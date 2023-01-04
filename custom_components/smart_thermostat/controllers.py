import abc
import logging
from datetime import timedelta, datetime
from homeassistant.util import dt as dt_util

from typing import Optional, final, Mapping, Any

from simple_pid import PID

from homeassistant.components.climate import DOMAIN as CLIMATE_DOMAIN
from homeassistant.components.climate import HVAC_MODE_OFF, HVAC_MODE_COOL, HVAC_MODE_HEAT, ATTR_HVAC_ACTION
from homeassistant.components.climate.const import CURRENT_HVAC_IDLE, SERVICE_SET_HVAC_MODE, ATTR_HVAC_MODE, \
    SERVICE_SET_TEMPERATURE, ATTR_MIN_TEMP, ATTR_MAX_TEMP, CURRENT_HVAC_OFF, ATTR_TARGET_TEMP_STEP
from homeassistant.components.input_number import ATTR_MIN, ATTR_MAX, SERVICE_SET_VALUE, ATTR_VALUE, ATTR_STEP
from homeassistant.const import STATE_OFF
from homeassistant.const import STATE_ON, ATTR_ENTITY_ID, SERVICE_TURN_ON, SERVICE_TURN_OFF, ATTR_TEMPERATURE
from homeassistant.core import DOMAIN as HA_DOMAIN, HomeAssistant, Context, CALLBACK_TYPE, State, split_entity_id
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition
from homeassistant.helpers.event import async_track_time_interval

_LOGGER = logging.getLogger(__name__)

ATTR_PID_PARAMS = "pid_params"

REASON_THERMOSTAT_STOP = "stop"
REASON_THERMOSTAT_FIRST_RUN = "first_run"
REASON_THERMOSTAT_HVAC_MODE_CHANGED = "hvac_mode_changed"
REASON_THERMOSTAT_TARGET_TEMP_CHANGED = "target_temp_changed"
REASON_THERMOSTAT_SENSOR_CHANGED = "sensor_changed"
REASON_THERMOSTAT_NOT_RUNNING = "not_running"
REASON_CONTROL_ENTITY_CHANGED = "control_entity_changed"
REASON_KEEP_ALIVE = "keep_alive"
REASON_PID_CONTROL = "pid_control"
REASON_PWM_CONTROL = "pwm_control"

PWM_SWITCH_ATTR_PWM_VALUE = "pwm_value"
PWM_SWITCH_ATTR_LAST_CONTROL_TIME = "last_control_time"
PWM_SWITCH_ATTR_LAST_CONTROL_STATE = "last_control_state"
PWM_SWITCH_MIN_VALUE = 0
PWM_SWITCH_MAX_VALUE = 100


def _round_step(value: float, step: float):
    return round(value / step) * step


class Thermostat(abc.ABC):
    @abc.abstractmethod
    def get_entity_id(self) -> str:
        """Get Entity name instance"""

    @abc.abstractmethod
    def get_context(self) -> Context:
        """Get Context instance"""

    @abc.abstractmethod
    def get_target_temperature(self):
        """Return the target temperature."""

    @abc.abstractmethod
    def get_current_temperature(self):
        """Return the sensor temperature."""

    @abc.abstractmethod
    def async_write_ha_state(self) -> None:
        """Write thermostat state."""

    @abc.abstractmethod
    def async_on_remove(self, func: CALLBACK_TYPE) -> None:
        """Add callback"""


class AbstractController(abc.ABC):
    """
    Abstract controller
    """

    def __init__(
            self,
            name: str,
            mode: str,
            target_entity_id: str,
            inverted: bool,
            keep_alive: Optional[timedelta]
    ):
        self._thermostat: Optional[Thermostat] = None
        self.name = name
        self._mode = mode
        self._target_entity_id = target_entity_id
        self._inverted = inverted
        self._keep_alive = keep_alive
        self.__running = False
        self._hass: Optional[HomeAssistant] = None
        if mode not in [HVAC_MODE_COOL, HVAC_MODE_HEAT]:
            raise ValueError(f"Unsupported mode: '{mode}'")

    def set_thermostat(self, thermostat: Thermostat):
        self._thermostat = thermostat

    @property
    @final
    def mode(self) -> str:
        return self._mode

    @property
    def _context(self) -> Context:
        return self._thermostat.get_context()

    @property
    @final
    def _thermostat_entity_id(self) -> str:
        return self._thermostat.get_entity_id()

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        return None

    async def async_added_to_hass(self, hass: HomeAssistant, attrs: Mapping[str, Any]):
        """Will be called in Entity async_added_to_hass()"""
        self._hass = hass

        if self._keep_alive:
            _LOGGER.info(
                "%s: %s - Setting up keep_alive: %s",
                self._thermostat_entity_id,
                self.name,
                self._keep_alive,
            )
            self._thermostat.async_on_remove(
                async_track_time_interval(
                    self._hass, self.__async_keep_alive, self._keep_alive
                )
            )

    @property
    def running(self):
        return self.__running

    @property
    @abc.abstractmethod
    def working(self):
        """Is target working now?"""

    def get_unique_id(self):
        """Get unique ID, for attrs storage"""
        name = "ctrl_" + split_entity_id(self._target_entity_id)[1]
        return name

    def get_used_entity_ids(self) -> [str]:
        """Get all used entity IDs to subscribe state change on them"""
        return [self._target_entity_id]

    @final
    async def async_start(self):
        cur_temp = self._thermostat.get_current_temperature()
        target_temp = self._thermostat.get_target_temperature()

        _LOGGER.debug(
            "%s: %s - Trying to start controller, cur: %s, target: %s ",
            self._thermostat_entity_id, self.name,
            cur_temp, target_temp
        )

        if await self._async_start(cur_temp, target_temp):
            self.__running = True
            _LOGGER.debug(
                "%s: %s - Started controller, cur: %s, target: %s ",
                self._thermostat_entity_id, self.name,
                cur_temp, target_temp
            )
        else:
            _LOGGER.error(
                "%s: %s - Error starting controller, cur: %s, target: %s ",
                self._thermostat_entity_id, self.name,
                cur_temp, target_temp
            )

    @final
    async def async_stop(self):
        _LOGGER.debug(
            "%s: %s - Stopping controller",
            self._thermostat_entity_id,
            self.name
        )
        await self._async_stop()
        self.__running = False

    @abc.abstractmethod
    async def _async_start(self, cur_temp, target_temp) -> bool:
        """Start controller implementation"""

    @abc.abstractmethod
    async def _async_stop(self):
        """Stop controller implementation"""

    @final
    async def async_control(self, time=None, force=False, reason=None):
        """Callback which will be called from Climate Entity"""

        cur_temp = self._thermostat.get_current_temperature()
        target_temp = self._thermostat.get_target_temperature()

        # _LOGGER.debug("%s: %s - Control: cur: %s, target: %s, force: %s, time: %s, (%s)",
        #               self._thermostat_entity_id, self.name,
        #               cur_temp, target_temp,
        #               force,
        #               True if time else False,
        #               reason
        #               )

        if not self.__running:
            await self._async_ensure_not_running()
        else:
            await self._async_control(cur_temp, target_temp, time=time, force=force, reason=reason)

    @final
    async def __async_keep_alive(self, time=None):
        await self.async_control(time=time, reason=REASON_KEEP_ALIVE)

    @abc.abstractmethod
    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        """Control method. Should be overwritten in child classes"""

    @abc.abstractmethod
    async def _async_ensure_not_running(self):
        """Ensure that target is off"""


class SwitchController(AbstractController):

    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            cold_tolerance: float,
            hot_tolerance: float,
            inverted: bool,
            keep_alive: Optional[timedelta],
            min_cycle_duration
    ):
        super().__init__(name, mode, target_entity_id, inverted, keep_alive)
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._min_cycle_duration = min_cycle_duration

    @property
    def working(self):
        return self._is_on()

    async def _async_turn_on(self, reason):
        _LOGGER.info("%s: %s - Turning on switch %s (%s)",
                     self._thermostat_entity_id,
                     self.name, self._target_entity_id, reason)

        service = SERVICE_TURN_ON if not self._inverted else SERVICE_TURN_OFF
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_turn_off(self, reason):
        _LOGGER.info("%s: %s - Turning off switch %s (%s)",
                     self._thermostat_entity_id,
                     self.name, self._target_entity_id, reason)

        service = SERVICE_TURN_OFF if not self._inverted else SERVICE_TURN_ON
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    def _is_on(self):
        return self._hass.states.is_state(
            self._target_entity_id,
            STATE_ON if not self._inverted else STATE_OFF
        )

    async def _async_start(self, cur_temp, target_temp) -> bool:
        return True

    async def _async_stop(self):
        await self._async_turn_off(reason=REASON_THERMOSTAT_STOP)

    async def _async_ensure_not_running(self):
        if self._is_on():
            await self._async_turn_off(REASON_THERMOSTAT_NOT_RUNNING)

    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        # If the `force` argument is True, we
        # ignore `min_cycle_duration`.
        if not force and reason == REASON_KEEP_ALIVE and self._min_cycle_duration:
            if self._is_on():
                current_state = STATE_ON
            else:
                current_state = HVAC_MODE_OFF
            try:
                long_enough = condition.state(
                    self._hass,
                    self._target_entity_id,
                    current_state,
                    self._min_cycle_duration,
                )
            except ConditionError:
                long_enough = False

            if not long_enough:
                return

        too_cold = cur_temp <= target_temp - self._cold_tolerance
        too_hot = cur_temp >= target_temp + self._hot_tolerance

        need_turn_on = (too_hot and self._mode == HVAC_MODE_COOL) or (too_cold and self._mode == HVAC_MODE_HEAT)

        _LOGGER.debug(f"%s: %s - too_hot: %s, too_cold: %s, need_turn_on: %s, is on: %s, cur: %s, target: %s (%s)",
                      self._thermostat_entity_id, self.name,
                      too_hot, too_cold,
                      need_turn_on, self._is_on(),
                      cur_temp, target_temp,
                      reason
                      )

        if self._is_on():
            if not need_turn_on:
                await self._async_turn_off(reason=reason)
            elif reason == REASON_KEEP_ALIVE:
                # The time argument is passed only in keep-alive case
                await self._async_turn_on(reason=reason)
        else:
            if need_turn_on:
                await self._async_turn_on(reason=reason)
            elif reason == REASON_KEEP_ALIVE:
                await self._async_turn_off(reason=reason)


class PidParams(abc.ABC):
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def invert(self):
        self.kp = -self.kp
        self.ki = -self.ki
        self.kd = -self.kd

    def __repr__(self):
        return f"Kp={self.kp}, Ki={self.ki}, Kd={self.kd}"


class AbstractPidController(AbstractController, abc.ABC):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            pid_sample_period: Optional[timedelta],
            inverted: bool,
            keep_alive: Optional[timedelta],
            output_min: Optional[float],
            output_max: Optional[float]
    ):
        super().__init__(name, mode, target_entity_id, inverted, keep_alive)
        self._initial_pid_params = pid_params
        self._current_pid_params: Optional[PidParams] = None
        self._pid_sample_period = pid_sample_period
        self._output_min = output_min
        self._output_max = output_max
        self._pid: Optional[PID] = None
        self._auto_tune = False
        self._last_output: Optional[float] = None
        self._last_output_limits: None
        self._last_current_value = None

    async def async_added_to_hass(self, hass: HomeAssistant, attrs: Mapping[str, Any]):
        await super().async_added_to_hass(hass, attrs)

        if self._auto_tune:
            saved_pid_params = attrs.get(ATTR_PID_PARAMS, None)
            if saved_pid_params:
                kp, ki, kd = saved_pid_params.split(',')
                self._current_pid_params = PidParams(kp, ki, kd)
                _LOGGER.info("%s: %s - restored last tuned PID params: %s",
                             self._thermostat_entity_id,
                             self.name,
                             self._current_pid_params
                             )
            raise NotImplementedError(f"Auto-tuning PID params unsupported now")
        else:
            if self._initial_pid_params:
                self.set_pid_params(self._initial_pid_params, reason="initial")

        if self._pid_sample_period:
            _LOGGER.info("%s: %s - Setting up PID regulator. Mode: static period (%s)",
                         self._thermostat_entity_id,
                         self.name,
                         self._pid_sample_period
                         )
            self._thermostat.async_on_remove(
                async_track_time_interval(
                    self._hass, self.__async_pid_control, self._pid_sample_period
                )
            )
        else:
            _LOGGER.info("%s: %s - Setting up PID regulator. Mode: Dynamic period on sensor changes",
                         self._thermostat_entity_id,
                         self.name
                         )

    @final
    async def __async_pid_control(self, time=None):
        if not self.running:
            return
        await self.async_control(time=time, reason=REASON_PID_CONTROL)

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        attrs = {}
        if self._current_pid_params:
            p = self._current_pid_params
            attrs[ATTR_PID_PARAMS] = f"{p.kp},{p.ki},{p.kd}"
        return attrs

    @final
    def set_pid_params(self, pid_params: PidParams, reason=None):
        """Set new PID params."""
        if not pid_params:
            raise ValueError(f"PID params can't be None ({reason})")

        if self._mode == HVAC_MODE_COOL:
            pid_params.invert()
            _LOGGER.info("%s: %s - Cooler mode. Inverting all PID params: %s (%s)",
                         self._thermostat_entity_id,
                         self.name,
                         pid_params, reason
                         )
        if self._inverted:
            pid_params.invert()
            _LOGGER.info("%s: %s - Target behavior inverted requested in config. Inverting all PID params: %s (%s)",
                         self._thermostat_entity_id,
                         self.name,
                         pid_params, reason
                         )

        self._current_pid_params = pid_params

        if self._pid:
            self._pid.Kp = pid_params.kp
            self._pid.Ki = pid_params.ki
            self._pid.Kd = pid_params.kd

        _LOGGER.info("%s: %s - New PID params: %s (%ss)",
                     self._thermostat_entity_id,
                     self.name,
                     self._current_pid_params,
                     reason
                     )

    @abc.abstractmethod
    def _is_on(self):
        """Is turned on"""

    async def _async_start(self, cur_temp, target_temp) -> bool:
        return self._setup_pid(cur_temp, target_temp)

    async def _async_stop(self):
        self._reset_pid()
        self._last_current_value = None

    @final
    def _setup_pid(self, cur_temp, target_temp):
        _ = cur_temp

        output_limits = self.__get_output_limits()

        if not self.__validate_output_limits(output_limits):
            return False

        pid_params = self._current_pid_params

        self._pid = PID(
            pid_params.kp, pid_params.ki, pid_params.kp,
            setpoint=target_temp,
            output_limits=output_limits,
            auto_mode=False,
            sample_time=self._pid_sample_period.total_seconds() if self._pid_sample_period else None
        )

        current_output = self._round_to_target_precision(self._get_current_output())
        if current_output:
            self._pid.set_auto_mode(enabled=True, last_output=current_output)
        else:
            self._pid.set_auto_mode(enabled=True)

        self._last_output_limits = output_limits

        _LOGGER.debug("%s: %s - Setup PID done. (params: %s, current_output: %s, limits: %s)",
                      self._thermostat_entity_id, self.name,
                      pid_params,
                      current_output, output_limits
                      )

        return True

    def _reset_pid(self):
        self._pid = None
        self._last_output = None

    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        if not self._pid:
            # This should really never happen
            _LOGGER.error("%s: %s - No PID", self._thermostat_entity_id, self.name)
            return

        if self._pid.setpoint != target_temp:
            _LOGGER.info("%s: %s - Target setpoint was changed from %s to %s (%s)",
                         self._thermostat_entity_id, self.name,
                         self._pid.setpoint, target_temp, reason
                         )
            self._pid.setpoint = target_temp

        output_limits = self.__get_output_limits()
        if self._last_output_limits != output_limits:
            _LOGGER.info("%s: %s - Output limits were changed from %s to %s (%s)",
                         self._thermostat_entity_id, self.name,
                         self._last_output_limits, output_limits, reason
                         )
            if not self.__validate_output_limits(output_limits):
                return
            self._pid.output_limits = output_limits

        current_output = self._round_to_target_precision(self._get_current_output())

        if self._last_output is not None and self._last_output != current_output:
            _LOGGER.info("%s: %s - Target was changed manually from %s to %s - restarting PID regulator (%s)",
                         self._thermostat_entity_id, self.name,
                         self._last_output, current_output, reason
                         )
            self._reset_pid()
            self._setup_pid(cur_temp, target_temp)

        if reason == REASON_KEEP_ALIVE and self._last_output:
            await self._apply_output(self._last_output)
        elif reason in (REASON_THERMOSTAT_SENSOR_CHANGED, REASON_PID_CONTROL):
            # Run PID only if static period configured or if sensor was changed.

            output = self._round_to_target_precision(float(self._pid(cur_temp)))

            p, i, d = self._pid.components

            if current_output != output:
                _LOGGER.debug("%s: %s - Current temp: %s -> %s, target: %s, limits: %s, adjusting from %s to %s (%s) (p:%f, i:%f, d:%f)",
                              self._thermostat_entity_id, self.name,
                              self._last_current_value, cur_temp, target_temp, output_limits,
                              current_output, output, reason,
                              p, i, d
                              )
                await self._apply_output(output)
            else:
                _LOGGER.debug("%s: %s - Current temp: %s -> %s, target: %s, limits: %s, no changes needed, output: %s (%s) (p:%f, i:%f, d:%f)",
                              self._thermostat_entity_id, self.name,
                              self._last_current_value, cur_temp, target_temp, output_limits,
                              current_output, reason,
                              p, i, d
                              )

            self._last_output = output
            self._last_current_value = cur_temp

    def __validate_output_limits(self, output_limits: (None, None)) -> bool:
        min_output, max_output = output_limits

        if None in (min_output, max_output):
            _LOGGER.error(
                "%s: %s - Invalid output limits: (%s, %s)",
                self._thermostat_entity_id, self.name,
                min_output, max_output
            )
            return False
        else:
            return True

    def __get_output_limits(self) -> (None, None):
        output_limits = self._get_output_limits()
        min_limit, max_limit = output_limits

        # Override min/max values if provided in config
        if self._output_min is not None:
            if min_limit is not None and self._output_min < min_limit:
                _LOGGER.warning("%s: %s - config min (%) < target min (%) - not adjusting",
                                self._thermostat_entity_id,
                                self.name,
                                self._output_min,
                                min_limit
                                )
            else:
                min_limit = self._output_min
        if self._output_max is not None:
            if max_limit is not None and self._output_max > max_limit:
                _LOGGER.warning("%s: %s - config max (%) > target max (%) - not adjusting",
                                self._thermostat_entity_id,
                                self.name,
                                self._output_max,
                                max_limit
                                )
            else:
                max_limit = self._output_max

        return min_limit, max_limit

    @abc.abstractmethod
    def _round_to_target_precision(self, value: float) -> float:
        """Round output to target precision"""

    @abc.abstractmethod
    def _get_current_output(self):
        """Get current output"""

    @abc.abstractmethod
    def _get_output_limits(self) -> (None, None):
        """Get output limits (min,max) in controller implementation"""

    @abc.abstractmethod
    async def _apply_output(self, output: float):
        """Apply output to target"""


class PwmSwitchPidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            pid_sample_period: timedelta,
            inverted: bool,
            keep_alive: Optional[timedelta],
            pwm_period: timedelta,
    ):
        super().__init__(name, mode, target_entity_id,
                         pid_params, pid_sample_period,
                         inverted, keep_alive,
                         None, None  # No config options. static values are PWM_SWITCH_MIN_VALUE/PWM_SWITCH_MAX_VALUE
                         )
        self._pwm_period = pwm_period
        self._pwm_value: Optional[int] = None
        target_entity_name = split_entity_id(target_entity_id)[1]
        self._pwm_value_attr_name = target_entity_name + PWM_SWITCH_ATTR_PWM_VALUE

        self._pwm_control_period = self._pwm_period / PWM_SWITCH_MAX_VALUE

        if self._pwm_control_period < timedelta(seconds=1):
            self._pwm_control_period = timedelta(seconds=1)

        self._last_control_time: Optional[datetime] = None
        self._last_control_state: Optional[str] = None

    async def async_added_to_hass(self, hass: HomeAssistant, attrs: Mapping[str, Any]):
        await super().async_added_to_hass(hass, attrs)

        pwm_value = attrs.get(PWM_SWITCH_ATTR_PWM_VALUE, None)
        if pwm_value is not None:
            self._pwm_value = self._round_to_target_precision(pwm_value)

        last_control_time = attrs.get(PWM_SWITCH_ATTR_LAST_CONTROL_TIME, None)
        if last_control_time is not None:
            self._last_control_time = dt_util.parse_datetime(last_control_time)

        self._last_control_state = attrs.get(PWM_SWITCH_ATTR_LAST_CONTROL_STATE, None)

        if self._pwm_value is None:
            # Apply default output value
            output = int((PWM_SWITCH_MIN_VALUE + PWM_SWITCH_MAX_VALUE) / 2)
            await self._apply_output(output)

        _LOGGER.info("%s: %s - Setting up PWM switch. PWM value: %s, period: %s, last control: [state: %s, time: %s], check PWM control every %s",
                     self._thermostat_entity_id,
                     self.name,
                     self._pwm_value,
                     self._pwm_period,
                     self._last_control_state,
                     self._last_control_time,
                     self._pwm_control_period
                     )
        self._thermostat.async_on_remove(
            async_track_time_interval(
                self._hass, self._async_pwm_control, self._pwm_control_period
            )
        )

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        attrs = super().extra_state_attributes or {}

        if None not in (self._last_control_time, self._last_control_state):
            attrs.update({
                PWM_SWITCH_ATTR_LAST_CONTROL_TIME: self._last_control_time.replace(microsecond=0),
                PWM_SWITCH_ATTR_LAST_CONTROL_STATE: self._last_control_state
            })

        if self._pwm_value is not None:
            attrs[PWM_SWITCH_ATTR_PWM_VALUE] = self._pwm_value

        return attrs

    async def _async_pwm_control(self, time=None):
        if not self.running:
            return
        await self.async_control(time=time, reason=REASON_PWM_CONTROL)

    @property
    def working(self):
        return self._is_on()

    def _is_on(self):
        return self._hass.states.is_state(
            self._target_entity_id,
            STATE_ON if not self._inverted else STATE_OFF
        )

    async def _async_turn_on(self, reason):
        _LOGGER.info("%s: %s - Turning on PWM switch %s (%s)",
                     self._thermostat_entity_id,
                     self.name, self._target_entity_id, reason)

        service = SERVICE_TURN_ON if not self._inverted else SERVICE_TURN_OFF
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_turn_off(self, reason):
        _LOGGER.info("%s: %s - Turning off PWM switch %s (%s)",
                     self._thermostat_entity_id,
                     self.name, self._target_entity_id, reason)

        service = SERVICE_TURN_OFF if not self._inverted else SERVICE_TURN_ON
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_stop(self):
        await self._async_turn_off(reason=REASON_THERMOSTAT_STOP)
        await super()._async_stop()

    async def _async_ensure_not_running(self):
        if self._is_on():
            await self._async_turn_off(REASON_THERMOSTAT_NOT_RUNNING)

    def _round_to_target_precision(self, value: float) -> float:
        # PWM value always int
        return int(value)

    def _get_output_limits(self) -> (None, None):
        return PWM_SWITCH_MIN_VALUE, PWM_SWITCH_MAX_VALUE

    def _get_current_output(self):
        return self._pwm_value

    async def _apply_output(self, output: float):
        if self._pwm_value != output:
            self._pwm_value = output
            self._thermostat.async_write_ha_state()

    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        await super()._async_control(cur_temp, target_temp, time, force, reason)

        if self._last_control_state:
            # Check real state is correct or keepalive requested
            if self._last_control_state == STATE_ON and (reason == REASON_KEEP_ALIVE or not self._is_on()):
                _LOGGER.debug("%s: %s Force ON (%s)", self._thermostat_entity_id, self.name, reason)
                await self._async_turn_on(reason=reason)
            elif self._last_control_state == STATE_OFF and (reason == REASON_KEEP_ALIVE or self._is_on()):
                _LOGGER.debug("%s: %s Force OFF (%s)", self._thermostat_entity_id, self.name, reason)
                await self._async_turn_off(reason=reason)

        elif self._is_on():
            # no _last_control_state - should be always off
            await self._async_turn_off(reason=reason)

        if reason == REASON_PWM_CONTROL:
            await self._pwm_control(reason=reason)

    async def _pwm_control(self, reason):
        if self._pwm_value is None:
            # This should really never happen
            _LOGGER.error("%s: %s - No PWM value (%s)", self._thermostat_entity_id, self.name, reason)
            return

        new_state = None  # will be applied if not None

        pwm_on_duration: timedelta = self._pwm_period * self._pwm_value / PWM_SWITCH_MAX_VALUE
        pwm_off_duration: timedelta = self._pwm_period - pwm_on_duration

        need_to_wait = None

        if None in (self._last_control_state, self._last_control_time):
            # Start with ON state
            new_state = STATE_ON
        else:
            now = dt_util.now().replace(microsecond=0)

            if self._last_control_state == STATE_ON and pwm_off_duration.total_seconds() > 0:
                if now >= (self._last_control_time + pwm_on_duration):
                    new_state = STATE_OFF
                else:
                    need_to_wait = self._last_control_time + pwm_on_duration - now

            elif self._last_control_state == STATE_OFF and pwm_on_duration.total_seconds() > 0:
                if now >= (self._last_control_time + pwm_off_duration):
                    new_state = STATE_ON
                else:
                    need_to_wait = self._last_control_time + pwm_off_duration - now

        change_info = f"`{self._last_control_state}` -> `{new_state}`" \
            if new_state \
            else f"`{self._last_control_state}` - wait {need_to_wait}"

        _LOGGER.debug("%s: %s - PWM value: %s, last: {state: %s, time: %s}, dur: {on: %s, off: %s}: state: {%s} ",
                      self._thermostat_entity_id, self.name,
                      self._pwm_value,
                      self._last_control_state,
                      self._last_control_time,
                      pwm_on_duration, pwm_off_duration,
                      change_info
                      )

        # Save last control params
        if new_state:
            self._last_control_time = dt_util.now().replace(microsecond=0)
            self._last_control_state = new_state
            self._thermostat.async_write_ha_state()

            if new_state == STATE_ON:
                await self._async_turn_on(reason=reason)
            elif new_state == STATE_OFF:
                await self._async_turn_off(reason=reason)


class NumberPidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            pid_sample_period: timedelta,
            inverted: bool,
            keep_alive: Optional[timedelta],
            output_min: Optional[float],
            output_max: Optional[float],
            switch_entity_id: str,
            switch_inverted: bool
    ):
        super().__init__(name, mode, target_entity_id,
                         pid_params, pid_sample_period,
                         inverted, keep_alive,
                         output_min, output_max)
        self._switch_entity_id = switch_entity_id
        self._switch_inverted = switch_inverted

    def get_used_entity_ids(self) -> [str]:
        ids = super().get_used_entity_ids()
        ids.append(self._switch_entity_id)
        return ids

    @property
    def working(self):
        # We don't have any information is real heater/cooler working now
        # So we just return switch state
        # TODO: Add optional binary sensor which will indicate real operating status of the target
        return self._is_on()

    def _is_on(self):
        return self._hass.states.is_state(
            self._switch_entity_id,
            STATE_ON if not self._switch_inverted else STATE_OFF
        )

    async def _async_turn_on(self, reason=None):
        _LOGGER.info("%s: %s - Turning on switch %s (%s)",
                     self._switch_entity_id,
                     self.name, self._switch_entity_id, reason)

        service = SERVICE_TURN_ON if not self._switch_inverted else SERVICE_TURN_OFF
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._switch_entity_id
        }, context=self._context)

    async def _async_turn_off(self, reason):
        _LOGGER.info("%s: %s - Turning off switch %s (%s)",
                     self._switch_entity_id,
                     self.name, self._switch_entity_id, reason)

        service = SERVICE_TURN_OFF if not self._switch_inverted else SERVICE_TURN_ON
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._switch_entity_id
        }, context=self._context)

    async def _async_stop(self):
        await super()._async_stop()
        await self._async_turn_off(REASON_THERMOSTAT_STOP)

    async def _async_ensure_not_running(self):
        if self._is_on():
            await self._async_turn_off(REASON_THERMOSTAT_NOT_RUNNING)

    def _round_to_target_precision(self, value: float) -> float:
        state: State = self._hass.states.get(self._target_entity_id)
        if not state:
            return value
        step = state.attributes.get(ATTR_STEP)
        return _round_step(value, step)

    def _get_current_output(self):
        state = self._hass.states.get(self._target_entity_id)
        if state:
            return float(state.state)
        return None

    def _get_output_limits(self):
        min_temp, max_temp = (None, None)

        state: State = self._hass.states.get(self._target_entity_id)
        if state:
            min_temp = state.attributes.get(ATTR_MIN)
            max_temp = state.attributes.get(ATTR_MAX)

        return min_temp, max_temp

    async def _apply_output(self, output: float):
        domain = split_entity_id(self._target_entity_id)[0]

        await self._hass.services.async_call(
            domain, SERVICE_SET_VALUE, {
                ATTR_ENTITY_ID: self._target_entity_id,
                ATTR_VALUE: output
            }, context=self._context
        )

    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        if not self._is_on() or reason == REASON_KEEP_ALIVE:
            await self._async_turn_on(reason=reason)
        await super()._async_control(cur_temp, target_temp, time, force, reason)


class ClimatePidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            pid_sample_period: timedelta,
            inverted: bool,
            keep_alive: Optional[timedelta],
            output_min: Optional[float],
            output_max: Optional[float]
    ):
        super().__init__(name, mode, target_entity_id,
                         pid_params, pid_sample_period,
                         inverted, keep_alive,
                         output_min, output_max)

    @property
    def working(self):
        state: State = self._hass.states.get(self._target_entity_id)
        if not state:
            return False
        hvac_action = state.attributes.get(ATTR_HVAC_ACTION)
        return hvac_action not in (CURRENT_HVAC_IDLE, CURRENT_HVAC_OFF)

    def _is_on(self):
        state: State = self._hass.states.get(self._target_entity_id)
        if not state:
            return False
        return state.state == self.mode

    def _round_to_target_precision(self, value: float) -> float:
        state: State = self._hass.states.get(self._target_entity_id)
        if not state:
            return value
        step = state.attributes.get(ATTR_TARGET_TEMP_STEP)
        return _round_step(value, step)

    def _get_current_output(self):
        state = self._hass.states.get(self._target_entity_id)
        if state:
            return float(state.attributes.get(ATTR_TEMPERATURE))
        return None

    async def _async_turn_on(self, reason=None):
        _LOGGER.debug("%s: %s - Setting HVAC mode to %s", self._thermostat_entity_id, self.name, self.mode)
        data = {
            ATTR_ENTITY_ID: self._target_entity_id,
            ATTR_HVAC_MODE: self.mode
        }
        await self._hass.services.async_call(
            CLIMATE_DOMAIN, SERVICE_SET_HVAC_MODE, data, context=self._context
        )

    async def _async_turn_off(self, reason):
        _LOGGER.info("%s: %s - Turning off climate %s (%s)",
                     self._thermostat_entity_id,
                     self.name, self._target_entity_id, reason)

        await self._hass.services.async_call(CLIMATE_DOMAIN, SERVICE_TURN_OFF, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_stop(self):
        await super()._async_stop()
        await self._async_turn_off(REASON_THERMOSTAT_STOP)

    async def _async_ensure_not_running(self):
        if self._is_on():
            await self._async_turn_off(REASON_THERMOSTAT_NOT_RUNNING)

    def _get_output_limits(self) -> (None, None):
        min_temp = None
        max_temp = None

        state: State = self._hass.states.get(self._target_entity_id)
        if state:
            min_temp = state.attributes.get(ATTR_MIN_TEMP)
            max_temp = state.attributes.get(ATTR_MAX_TEMP)

        return min_temp, max_temp

    async def _apply_output(self, output: float):
        await self._hass.services.async_call(
            CLIMATE_DOMAIN, SERVICE_SET_TEMPERATURE, {
                ATTR_ENTITY_ID: self._target_entity_id,
                ATTR_TEMPERATURE: output
            }, context=self._context
        )

    async def _async_control(self, cur_temp, target_temp, time=None, force=False, reason=None):
        if not self._is_on() or reason == REASON_KEEP_ALIVE:
            await self._async_turn_on(reason=reason)
        await super()._async_control(cur_temp, target_temp, time, force, reason)
