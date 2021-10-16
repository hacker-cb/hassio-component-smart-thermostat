import abc
import logging
from typing import Optional, final, Mapping, Any

from homeassistant.components.climate import HVAC_MODE_OFF, HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_HEAT_COOL
from homeassistant.const import STATE_ON, ATTR_ENTITY_ID, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import DOMAIN as HA_DOMAIN, callback, Event, HomeAssistant, Context, CALLBACK_TYPE, State
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition
from homeassistant.helpers.event import async_track_state_change_event

ATTR_PID_PARAMS = "pid_params"

_LOGGER = logging.getLogger(__name__)


class Thermostat(abc.ABC):
    @abc.abstractmethod
    def get_hass(self) -> HomeAssistant:
        """Get HomeAssistant instance"""

    @abc.abstractmethod
    def get_entity_id(self) -> str:
        """Get Entity name instance"""

    @abc.abstractmethod
    def get_hvac_mode(self) -> str:
        """Get Current HVAC mode"""

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
    def async_write_ha_state(self):
        """Write thermostat state."""

    @abc.abstractmethod
    def async_on_remove(self, func: CALLBACK_TYPE) -> None:
        """Add callback"""

    @abc.abstractmethod
    async def async_get_last_state(self) -> Optional[State]:
        """Get last saved sate"""


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
        self._active = False
        if mode not in [HVAC_MODE_COOL, HVAC_MODE_HEAT]:
            raise ValueError(f"Unsupported mode: '{mode}'")

    def set_thermostat(self, thermostat: Thermostat):
        self._thermostat = thermostat

    @property
    def _hass(self) -> HomeAssistant:
        return self._thermostat.get_hass()

    @property
    def _hvac_mode(self) -> str:
        return self._thermostat.get_hvac_mode()

    @property
    def _context(self) -> Context:
        return self._thermostat.get_context()

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        return None

    async def async_added_to_hass(self):
        """Will be called in Entity async_added_to_hass()"""
        self._thermostat.async_on_remove(
            async_track_state_change_event(
                self._hass, [self._target_entity_id], self._on_target_entity_state_changed
            )
        )
        self._thermostat.async_write_ha_state()

    def async_startup(self):
        """
        Startup method. Will ve called after HA core started
        """
        self._hass.create_task(self.async_control())

    @callback
    def _on_target_entity_state_changed(self, event: Event):
        """On state changed callback"""
        _ = event
        self._hass.create_task(self.async_control())

    @property
    @abc.abstractmethod
    def running(self):
        """Is target running now?"""

    @property
    def _target_entity_state(self):
        """Get target state"""
        return self._hass.states.get(self._target_entity_id)

    @property
    def _allow_cool(self):
        return self._mode == HVAC_MODE_COOL and self._hvac_mode in [HVAC_MODE_COOL, HVAC_MODE_HEAT_COOL]

    @property
    def _allow_heat(self):
        return self._mode == HVAC_MODE_HEAT and self._hvac_mode in [HVAC_MODE_HEAT, HVAC_MODE_HEAT_COOL]

    @final
    async def async_control(self, time=None, force=False):
        """Callback which will be called from Climate Entity"""
        cur_temp = self._thermostat.get_current_temperature()
        target_temp = self._thermostat.get_target_temperature()
        await self._async_control(cur_temp, target_temp, time=time, force=force)

    @abc.abstractmethod
    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        """Control method. Should be overwritten in child classes"""


class PidParams(abc.ABC):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


class AbstractPidController(AbstractController, abc.ABC):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            inverted: bool
    ):
        super().__init__(name, mode, target_entity_id, inverted)
        self._initial_pid_params = pid_params
        self._current_pid_params = Optional[PidParams]

    async def async_added_to_hass(self):
        await super().async_added_to_hass()

        old_state = await self._thermostat.async_get_last_state()

        if old_state is not None and old_state.attributes.get(ATTR_PID_PARAMS) is not None:
            saved_pid_params = old_state.attributes.get(ATTR_PID_PARAMS)
            if saved_pid_params:
                kp, ki, kd = saved_pid_params.split(',')
                self._current_pid_params = PidParams(kp, ki, kd)
                _LOGGER.info("%s: %s - restored last PID params: %s",
                             self._thermostat.get_entity_id,
                             self.name,
                             self._current_pid_params
                             )
        if not self._current_pid_params:
            self._current_pid_params = self._initial_pid_params
            _LOGGER.info("%s: %s - No PID params found in state attributes, using default: %s",
                         self._thermostat.get_entity_id,
                         self.name,
                         self._current_pid_params if self._current_pid_params else None
                         )

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        p = self._current_pid_params
        return {
            ATTR_PID_PARAMS: f"{p.kp},{p.ki},{p.kd}" if p else None
        }


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

    @AbstractController.running.getter
    def running(self):
        """If the toggleable device is currently active."""
        if not self._target_entity_state:
            return None

        return self._hass.states.is_state(self._target_entity_id, STATE_ON)

    async def _async_turn_on(self):
        """Turn toggleable device on."""
        service = SERVICE_TURN_ON if not self._inverted else SERVICE_TURN_OFF
        data = {ATTR_ENTITY_ID: self._target_entity_id}
        await self._hass.services.async_call(
            HA_DOMAIN, service, data, context=self._context
        )

    async def _async_turn_off(self):
        """Turn toggleable device off."""
        service = SERVICE_TURN_OFF if not self._inverted else SERVICE_TURN_ON
        data = {ATTR_ENTITY_ID: self._target_entity_id}
        await self._hass.services.async_call(
            HA_DOMAIN, service, data, context=self._context
        )

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        if not self._active and None not in (
                cur_temp,
                target_temp,
        ):
            self._active = True
            _LOGGER.info(
                "%s: %s - obtained current (%s) and target  (%s) temperature. "
                "Smart thermostat running.",
                self._thermostat.get_entity_id(),
                self.name,
                cur_temp,
                target_temp,
            )

        if self._hvac_mode == HVAC_MODE_OFF and self.running:
            _LOGGER.info("%s: %s - HVAC mode is off, but running, turning off",
                         self._thermostat.get_entity_id(),
                         self.name,
                         )
            await self._async_turn_off()

        if not self._active or self._hvac_mode == HVAC_MODE_OFF:
            return

        # If the `force` argument is True, we
        # ignore `min_cycle_duration`.
        # If the `time` argument is not none, we were invoked for
        # keep-alive purposes, and `min_cycle_duration` is irrelevant.
        if not force and time is None and self._min_cycle_duration:
            if self.running:
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

        need_run = False

        if self._allow_cool and too_hot:
            need_run = True
        elif self._allow_heat and too_cold:
            need_run = True

        _LOGGER.debug(f"%s: %s - too_hot: %s, too_cold: %s, need_run: %s (cur: %s, target: %s)",
                      self._thermostat.get_entity_id(),
                      self.name,
                      too_hot,
                      too_cold,
                      need_run,
                      cur_temp,
                      target_temp
                      )

        if self.running:
            if not need_run:
                _LOGGER.info("%s: Turning off %s %s",
                             self._thermostat.get_entity_id(),
                             self.name, self._target_entity_id)
                await self._async_turn_off()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info("%s: Keep-alive - Turning on %s %s",
                             self._thermostat.get_entity_id(),
                             self.name, self._target_entity_id)
                await self._async_turn_on()
        else:
            if need_run:
                _LOGGER.info("%s: Turning on %s %s",
                             self._thermostat.get_entity_id(),
                             self.name, self._target_entity_id)
                await self._async_turn_on()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info(
                    "%s: Keep-alive - Turning off %s %s",
                    self._thermostat.get_entity_id(),
                    self.name, self._target_entity_id
                )
                await self._async_turn_off()


class ClimatePidController(AbstractPidController):
    def __init__(
            self,
            name: str,
            mode,
            target_entity_id: str,
            pid_params: PidParams,
            inverted: bool
    ):
        super().__init__(name, mode, target_entity_id, pid_params, inverted)

    @AbstractController.running.getter
    def running(self):
        raise NotImplementedError()  # FIXME: Not implemented

    async def _async_control(self, cur_temp, target_temp, time=None, force=False):
        raise NotImplementedError()  # FIXME: Not implemented
