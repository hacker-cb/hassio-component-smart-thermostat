import abc
import logging
from typing import List, Optional

from custom_components.smart_thermostat.config import CONF_INVERTED, CONF_MIN_DUR
from homeassistant.components.climate import HVAC_MODE_OFF, HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_HEAT_COOL
from homeassistant.const import STATE_ON, ATTR_ENTITY_ID, SERVICE_TURN_ON, SERVICE_TURN_OFF, CONF_ENTITY_ID, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import DOMAIN as HA_DOMAIN, callback, HomeAssistant, Context
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition

_LOGGER = logging.getLogger(__name__)


class AbstractController(abc.ABC):
    """
    Abstract controller
    """

    def __init__(
            self,
            name: str,
            mode: str,
            target: {}
    ):
        self._name = name
        self._hass = Optional[HomeAssistant]
        self._context = Optional[Context]
        self._mode = mode
        self._hvac_mode = HVAC_MODE_OFF
        self._target = target
        self._target_entity_id = target[CONF_ENTITY_ID]
        self._active = False
        if mode not in [HVAC_MODE_COOL, HVAC_MODE_HEAT]:
            raise ValueError(f"Unsupported mode: '{mode}'")

    def set_hass(self, hass: HomeAssistant):
        self._hass = hass

    def set_context(self, context: Context):
        self._context = context

    def set_hvac_mode(self, hvac_mode: str):
        self._hvac_mode = hvac_mode

    @abc.abstractmethod
    def startup(self):
        """
        Startup method. Will ve called from `async_added_to_hass()`
        """

    def get_entities_to_subscribe_state_changes(self):
        return [self._target_entity_id]

    @callback
    @abc.abstractmethod
    def on_state_changed(self, event):
        """On state changed callback"""

    @property
    @abc.abstractmethod
    def running(self):
        """Is target running now?"""

    @property
    def _allow_cool(self):
        return self._hvac_mode in [HVAC_MODE_COOL, HVAC_MODE_HEAT_COOL]

    @property
    def _allow_heat(self):
        return self._hvac_mode in [HVAC_MODE_HEAT, HVAC_MODE_HEAT_COOL]


    @abc.abstractmethod
    async def async_control(self, cur_temp, target_temp, time=None, force=False):
        """Control method. Should be overwritten in child classes"""


class SwitchController(AbstractController):

    def __init__(
            self,
            name: str,
            mode,
            target: {},
            cold_tolerance,
            hot_tolerance
    ):
        super().__init__(name, mode, target)
        self.name = name
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._target_inverted = self._target[CONF_INVERTED]
        self._min_cycle_duration = self._target[CONF_MIN_DUR] if CONF_MIN_DUR in self._target else None

    @AbstractController.running.getter
    def running(self):
        """If the toggleable device is currently active."""
        if not self._hass.states.get(self._target_entity_id):
            return None

        return self._hass.states.is_state(self._target_entity_id, STATE_ON)

    @callback
    def on_state_changed(self, event):
        """Handle switch state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")
        if new_state is None:
            return
        if old_state is None:
            self._hass.create_task(self._check_switch_initial_state())

    async def _check_switch_initial_state(self):
        """Prevent the device from keep running if HVAC_MODE_OFF."""
        if self._hvac_mode == HVAC_MODE_OFF and self.running:
            _LOGGER.warning(
                "The climate mode is OFF, but the switch device is ON. Turning off device %s",
                self._target_entity_id,
            )
            await self._async_turn_off()

    def startup(self):
        switch_state = self._hass.states.get(self._target_entity_id)
        if switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
        ):
            self._hass.create_task(self._check_switch_initial_state())

    async def _async_turn_on(self):
        """Turn toggleable device on."""
        service = SERVICE_TURN_ON if not self._target_inverted else SERVICE_TURN_OFF
        data = {ATTR_ENTITY_ID: self._target_entity_id}
        await self._hass.services.async_call(
            HA_DOMAIN, service, data, context=self._context
        )

    async def _async_turn_off(self):
        """Turn toggleable device off."""
        service = SERVICE_TURN_OFF if not self._target_inverted else SERVICE_TURN_ON
        data = {ATTR_ENTITY_ID: self._target_entity_id}
        await self._hass.services.async_call(
            HA_DOMAIN, service, data, context=self._context
        )

    async def async_control(self, cur_temp, target_temp, time=None, force=False):
        if not self._active and None not in (
                cur_temp,
                target_temp,
        ):
            self._active = True
            _LOGGER.info(
                "Obtained current and target temperature. "
                "Smart thermostat running (%s). cur: %s, target: %s",
                self.name,
                cur_temp,
                target_temp,
            )

        if self._hvac_mode == HVAC_MODE_OFF and self.running:
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

        allow_cool = self._allow_cool and too_hot
        allow_heat = self._allow_heat and too_cold

        _LOGGER.debug(f"{self.name} ({self._mode}, current: {self._hvac_mode}): "
                      f"too_hot: {too_hot}, too_cold: {too_cold} "
                      f"allow_cool: {allow_cool}, allow_heat: {too_cold} "
                      f"cur: {cur_temp}, target: {target_temp}")

        if self.running:
            if (self._mode == HVAC_MODE_COOL and not allow_cool) or (self._mode == HVAC_MODE_HEAT and not allow_heat):
                _LOGGER.info("Turning off %s %s", self.name, self._target_entity_id)
                await self._async_turn_off()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info("Keep-alive - Turning on %s %s", self.name, self._target_entity_id)
                await self._async_turn_on()
        else:
            if (self._mode == HVAC_MODE_COOL and allow_cool) or (self._mode == HVAC_MODE_HEAT and allow_heat):
                _LOGGER.info("Turning on %s %s", self.name, self._target_entity_id)
                await self._async_turn_on()
            elif time is not None:
                # The time argument is passed only in keep-alive case
                _LOGGER.info(
                    "Keep-alive - Turning off %s %s", self.name, self._target_entity_id
                )
                await self._async_turn_off()
