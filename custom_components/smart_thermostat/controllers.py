import abc
import logging
from datetime import timedelta
from typing import Optional, final, Mapping, Any

from simple_pid import PID

from homeassistant.components.climate import DOMAIN as CLIMATE_DOMAIN
from homeassistant.components.climate import HVAC_MODE_OFF, HVAC_MODE_COOL, HVAC_MODE_HEAT, ATTR_HVAC_ACTION
from homeassistant.components.climate.const import CURRENT_HVAC_IDLE, SERVICE_SET_HVAC_MODE, ATTR_HVAC_MODE, \
    SERVICE_SET_TEMPERATURE, ATTR_MIN_TEMP, ATTR_MAX_TEMP, CURRENT_HVAC_OFF
from homeassistant.components.input_number import ATTR_MIN, ATTR_MAX, SERVICE_SET_VALUE, ATTR_VALUE
from homeassistant.const import STATE_OFF
from homeassistant.const import STATE_ON, ATTR_ENTITY_ID, SERVICE_TURN_ON, SERVICE_TURN_OFF, ATTR_TEMPERATURE
from homeassistant.core import DOMAIN as HA_DOMAIN, HomeAssistant, Context, CALLBACK_TYPE, State, split_entity_id
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition
from homeassistant.helpers.event import async_track_time_interval

ATTR_PID_PARAMS = "pid_params"

_LOGGER = logging.getLogger(__name__)


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
            inverted: bool
    ):
        self._thermostat: Optional[Thermostat] = None
        self.name = name
        self._mode = mode
        self._target_entity_id = target_entity_id
        self._inverted = inverted
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

    async def async_added_to_hass(self, hass: HomeAssistant, old_state: State):
        """Will be called in Entity async_added_to_hass()"""
        self._hass = hass

    @property
    def running(self):
        return self.__running

    @property
    @abc.abstractmethod
    def working(self):
        """Is target working now?"""

    def get_used_entity_ids(self) -> [str]:
        """Get all used entity IDs to subscribe state change on them"""
        return [self._target_entity_id]

    @final
    async def async_start(self):
        cur_temp = self._thermostat.get_current_temperature()
        target_temp = self._thermostat.get_target_temperature()

        _LOGGER.debug(
            "%s: %s - Trying to start controller, cur: %, target: %s "
            "Activated",
            self._thermostat_entity_id,
            self.name,
            cur_temp,
            target_temp,
        )

        if await self._async_start(cur_temp, target_temp):
            _LOGGER.debug(
                "%s: %s - Started controller, cur: %, target: %s "
                "Activated",
                self._thermostat_entity_id,
                self.name,
                cur_temp,
                target_temp,
            )
            self.__running = True
        else:
            _LOGGER.error(
                "%s: %s - Error starting controller, cur: %, target: %s "
                "Activated",
                self._thermostat_entity_id,
                self.name,
                cur_temp,
                target_temp,
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
    async def async_control(self, time=None, force=False):
        """Callback which will be called from Climate Entity"""
        if not self.__running:
            return

        cur_temp = self._thermostat.get_current_temperature()
        target_temp = self._thermostat.get_target_temperature()

        _LOGGER.debug("%s: %s - Control: cur: %s, target: %s",
                      self._thermostat_entity_id,
                      self.name,
                      cur_temp,
                      target_temp
                      )

        await self._async_control(cur_temp, target_temp, time=time, force=force)

    @abc.abstractmethod
    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        """Control method. Should be overwritten in child classes"""


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
            inverted: bool,
            sample_period: timedelta,
            target_min: Optional[float],
            target_max: Optional[float]
    ):
        super().__init__(name, mode, target_entity_id, inverted)
        self._initial_pid_params = pid_params
        self._current_pid_params: Optional[PidParams] = None
        self._sample_period = sample_period
        self._target_min = target_min
        self._target_max = target_max
        self._pid: Optional[PID] = None
        self._auto_tune = False
        self._last_output: Optional[float] = None
        self._last_output_limits: None

    @final
    async def async_added_to_hass(self, hass: HomeAssistant, old_state: State):
        await super().async_added_to_hass(hass, old_state)

        if self._auto_tune:
            if old_state is not None and old_state.attributes.get(ATTR_PID_PARAMS) is not None:
                saved_pid_params = old_state.attributes.get(ATTR_PID_PARAMS)
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
                _LOGGER.info("%s: %s - using initial PID params: %s",
                             self._thermostat_entity_id,
                             self.name,
                             self._initial_pid_params
                             )
                self.set_pid_params(self._initial_pid_params)

        self._thermostat.async_on_remove(
            async_track_time_interval(
                self._hass, self.async_control, self._sample_period
            )
        )

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        if not self._auto_tune:
            return None

        p = self._current_pid_params
        return {
            ATTR_PID_PARAMS: f"{p.kp},{p.ki},{p.kd}" if p else None
        }

    @final
    def set_pid_params(self, pid_params: PidParams):
        """Set new PID params."""
        if not pid_params:
            raise ValueError(f"PID params can't be None")

        if self._mode == HVAC_MODE_COOL and not pid_params.kp < 0:
            pid_params.invert()
            _LOGGER.warning("%s: %s - Cooler mode but kp not negative. Inverting all PID params: %s",
                            self._thermostat_entity_id,
                            self.name,
                            pid_params
                            )
        if self._inverted:
            pid_params.invert()
            _LOGGER.info("%s: %s - Target behavior inverted requested in config. Inverting all PID params: %s",
                         self._thermostat_entity_id,
                         self.name,
                         pid_params
                         )

        self._current_pid_params = pid_params

        if self._pid:
            self._pid.Kp = pid_params.kp
            self._pid.Ki = pid_params.ki
            self._pid.Kd = pid_params.kd

        _LOGGER.info("%s: %s - New PID params: %s",
                     self._thermostat_entity_id,
                     self.name,
                     self._current_pid_params
                     )

    @final
    async def _async_start(self, cur_temp, target_temp) -> bool:

        if not self._current_pid_params:
            _LOGGER.error("%s: %s - Start called but no PID params was set", self._thermostat_entity_id, self.name)
            return False

        output_limits = self._get_output_limits()

        if not self.__validate_output_limits(output_limits):
            return False

        pid_params = self._current_pid_params

        self._pid = PID(
            pid_params.kp, pid_params.ki, pid_params.kp,
            setpoint=target_temp,
            output_limits=output_limits,
            auto_mode=False,
            sample_time=self._sample_period.total_seconds()
        )

        current_output = self.__round_to_target_precision(self._get_current_output())
        if current_output:
            self._pid.set_auto_mode(enabled=True, last_output=current_output)
        else:
            self._pid.set_auto_mode(enabled=True)

        _LOGGER.info("%s: %s - Initialized.  PID params: %s, temperature: %s, limits: %s",
                     self._thermostat_entity_id,
                     self.name,
                     pid_params,
                     current_output,
                     output_limits
                     )

        await self._async_turn_on()

        self._last_output_limits = output_limits

        return True

    @final
    async def _async_stop(self):
        await self._async_turn_off()
        self._pid = None
        pass

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        if not self._pid:
            _LOGGER.error("%s: %s - No PID instance to control", self._thermostat_entity_id, self.name)
            return False

        if self._pid.setpoint != target_temp:
            _LOGGER.info("%s: %s - Target setpoint was changed from %s to %s",
                         self._thermostat_entity_id,
                         self.name,
                         self._pid.setpoint,
                         target_temp
                         )
            self._pid.setpoint = target_temp

        output_limits = self._get_output_limits()
        if self._last_output_limits != output_limits:
            _LOGGER.info("%s: %s - Output limits were changed from %s to %s",
                         self._thermostat_entity_id,
                         self.name,
                         self._last_output_limits,
                         output_limits
                         )
            if not self.__validate_output_limits(output_limits):
                return
            self._pid.output_limits = output_limits

        temperature = self.__round_to_target_precision(self._get_current_output())

        if self._last_output is not None and self._last_output != temperature:
            _LOGGER.info("%s: %s - Target was changed manually from %s to %s - restarting PID regulator",
                         self._thermostat_entity_id,
                         self.name,
                         self._last_output,
                         temperature
                         )
            await self._async_start(cur_temp, target_temp)

        output = self.__round_to_target_precision(float(self._pid(cur_temp)))

        if temperature != output:
            _LOGGER.debug("%s: %s - Current temp: %s, target temp: %s, adjusting from %s to %s",
                          self._thermostat_entity_id,
                          self.name,
                          cur_temp,
                          target_temp,
                          temperature,
                          output
                          )
            await self._apply_output(output)

        self._last_output = output

    def __validate_output_limits(self, output_limits: (None, None)) -> bool:
        min_temp, max_temp = output_limits

        if not min_temp or not max_temp:
            _LOGGER.error(
                "%s: %s - Invalid output limits: (%s, %s)",
                self._thermostat_entity_id, self.name,
                min_temp, max_temp
            )
            return False
        else:
            return True

    def _get_output_limits(self) -> (None, None):
        output_limits = self._get_target_output_limits()
        min_temp, max_temp = output_limits

        # Override min/max values if provided in config
        if self._target_min is not None:
            if min_temp is not None and self._target_min < min_temp:
                _LOGGER.warning("%s: %s - config min (%) < target min (%) - not adjusting",
                                self._thermostat_entity_id,
                                self.name,
                                self._target_min,
                                min_temp
                                )
            else:
                min_temp = self._target_min
        if self._target_max is not None:
            if max_temp is not None and self._target_max > max_temp:
                _LOGGER.warning("%s: %s - config max (%) > target max (%) - not adjusting",
                                self._thermostat_entity_id,
                                self.name,
                                self._target_max,
                                max_temp
                                )
            else:
                max_temp = self._target_max

        return min_temp, max_temp

    def __round_to_target_precision(self, value: float) -> float:
        # FIXME: use target attr precision
        return round(value, 1)

    @abc.abstractmethod
    def _get_current_output(self):
        """Get current output"""

    @abc.abstractmethod
    async def _async_turn_on(self):
        """Turn on target"""

    @abc.abstractmethod
    async def _async_turn_off(self):
        """Turn off target"""

    @abc.abstractmethod
    def _get_target_output_limits(self) -> (None, None):
        """Get output limits (min,max) in controller implementation"""

    @abc.abstractmethod
    async def _apply_output(self, output: float):
        """Apply output to target"""


class SwitchController(AbstractController):

    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            cold_tolerance: float,
            hot_tolerance: float,
            inverted: bool,
            min_cycle_duration
    ):
        super().__init__(name, mode, target_entity_id, inverted)
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._min_cycle_duration = min_cycle_duration

    @property
    def working(self):
        return self._is_on()

    async def _async_turn_on(self):
        """Turn toggleable device on."""
        service = SERVICE_TURN_ON if not self._inverted else SERVICE_TURN_OFF
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_turn_off(self):
        """Turn toggleable device off."""
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
        await self._async_turn_off()

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        # If the `force` argument is True, we
        # ignore `min_cycle_duration`.
        # If the `time` argument is not none, we were invoked for
        # keep-alive purposes, and `min_cycle_duration` is irrelevant.
        if not force and time is None and self._min_cycle_duration:
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

        _LOGGER.debug(f"%s: %s - too_hot: %s, too_cold: %s, need_turn_on: %s, is on: %s, (cur: %s, target: %s)",
                      self._thermostat_entity_id,
                      self.name,
                      too_hot,
                      too_cold,
                      need_turn_on,
                      self._is_on(),
                      cur_temp,
                      target_temp
                      )

        if self._is_on():
            if not need_turn_on:
                _LOGGER.info("%s: %s - Turning off %s",
                             self._thermostat_entity_id,
                             self.name, self._target_entity_id)
                await self._async_turn_off()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info("%s: %s - Keep-alive - Turning on %s",
                             self._thermostat_entity_id,
                             self.name, self._target_entity_id)
                await self._async_turn_on()
        else:
            if need_turn_on:
                _LOGGER.info("%s: %s - Turning on %s",
                             self._thermostat_entity_id,
                             self.name, self._target_entity_id)
                await self._async_turn_on()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info(
                    "%s: %s - Keep-alive - Turning off %s",
                    self._thermostat_entity_id,
                    self.name, self._target_entity_id
                )
                await self._async_turn_off()


class NumberPidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            inverted: bool,
            sample_period: timedelta,
            target_min: Optional[float],
            target_max: Optional[float],
            switch_entity_id: Optional[str],
            switch_inverted: bool
    ):
        super().__init__(name, mode, target_entity_id,
                         pid_params, inverted,
                         sample_period,
                         target_min, target_max)
        self._switch_entity_id = switch_entity_id
        self._switch_inverted = switch_inverted

    def get_used_entity_ids(self) -> [str]:
        ids = super().get_used_entity_ids()
        if self._switch_entity_id:
            ids.append(self._switch_entity_id)
        return ids

    @property
    def working(self):
        return self._is_on()

    def _is_on(self):
        if not self._switch_entity_id:
            # FIXME: not good behavior, may be need to make switch required?
            return True  # Always working
        return self._hass.states.is_state(
            self._switch_entity_id,
            STATE_ON if not self._switch_inverted else STATE_OFF
        )

    async def _async_turn_on(self):
        if not self._switch_entity_id:
            return

        _LOGGER.debug("%s: %s - Turning on switch %s",
                      self._thermostat_entity_id,
                      self.name, self._switch_entity_id)

        service = SERVICE_TURN_ON if not self._switch_inverted else SERVICE_TURN_OFF
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._switch_entity_id
        }, context=self._context)

    async def _async_turn_off(self):
        if not self._switch_entity_id:
            return

        _LOGGER.debug("%s: %s - Turning off switch %s",
                      self._thermostat_entity_id,
                      self.name, self._switch_entity_id)

        service = SERVICE_TURN_OFF if not self._switch_inverted else SERVICE_TURN_ON
        await self._hass.services.async_call(HA_DOMAIN, service, {
            ATTR_ENTITY_ID: self._switch_entity_id
        }, context=self._context)

    def _get_current_output(self):
        state = self._hass.states.get(self._target_entity_id)
        if state:
            return float(state.state)

    def _get_target_output_limits(self):
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

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        if not self._is_on():
            await self._async_turn_on()

        await super()._async_control(cur_temp, target_temp, time, force)


class ClimatePidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            inverted: bool,
            sample_period: timedelta,
            target_min: Optional[float],
            target_max: Optional[float]
    ):
        super().__init__(name, mode, target_entity_id,
                         pid_params, inverted,
                         sample_period,
                         target_min, target_max)

    @property
    def working(self):
        state: State = self._hass.states.get(self._target_entity_id)
        if not state:
            return False
        hvac_action = state.attributes.get(ATTR_HVAC_ACTION)
        return hvac_action not in (CURRENT_HVAC_IDLE, CURRENT_HVAC_OFF)

    def _get_current_output(self):
        state = self._hass.states.get(self._target_entity_id)
        if state:
            return float(state.attributes.get(ATTR_TEMPERATURE))
        return None

    async def _async_turn_on(self):
        await self._hass.services.async_call(CLIMATE_DOMAIN, SERVICE_TURN_ON, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    async def _async_turn_off(self):
        await self._hass.services.async_call(CLIMATE_DOMAIN, SERVICE_TURN_OFF, {
            ATTR_ENTITY_ID: self._target_entity_id
        }, context=self._context)

    def _get_target_output_limits(self) -> (None, None):
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

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        state: State = self._hass.states.get(self._target_entity_id)
        if state:
            new_hvac_mode = None

            if self._mode == HVAC_MODE_COOL and state.state != HVAC_MODE_COOL:
                new_hvac_mode = HVAC_MODE_COOL
            elif self._mode == HVAC_MODE_HEAT and state.state != HVAC_MODE_HEAT:
                new_hvac_mode = HVAC_MODE_HEAT

            if new_hvac_mode:
                _LOGGER.debug(
                    "%s: %s - Setting HVAC mode to %s",
                    self._thermostat_entity_id,
                    self.name,
                    new_hvac_mode
                )
                data = {
                    ATTR_ENTITY_ID: self._target_entity_id,
                    ATTR_HVAC_MODE: new_hvac_mode
                }
                await self._hass.services.async_call(
                    CLIMATE_DOMAIN, SERVICE_SET_HVAC_MODE, data, context=self._context
                )

        # call parent
        await super()._async_control(cur_temp, target_temp, time, force)
