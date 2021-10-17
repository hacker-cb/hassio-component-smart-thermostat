"""
Adds support for smart thermostat units.
forked from HA-core `generic_thermostat` 827501659c926ace3741425760b1294d2e93b48e
"""
import asyncio
import logging
import math
from typing import Mapping, Any, Optional

import voluptuous as vol
from voluptuous import ALLOW_EXTRA

from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate.const import (
    ATTR_PRESET_MODE,
    CURRENT_HVAC_COOL,
    CURRENT_HVAC_HEAT,
    CURRENT_HVAC_IDLE,
    CURRENT_HVAC_OFF,
    HVAC_MODE_COOL,
    HVAC_MODE_HEAT,
    HVAC_MODE_HEAT_COOL,
    HVAC_MODE_OFF,
    PRESET_AWAY,
    PRESET_NONE,
    SUPPORT_PRESET_MODE,
    SUPPORT_TARGET_TEMPERATURE
)
from homeassistant.components.input_boolean import DOMAIN as INPUT_BOOLEAN_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.climate import DOMAIN as CLIMATE_DOMAIN
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    PRECISION_TENTHS,
    PRECISION_HALVES,
    PRECISION_WHOLE
)
from homeassistant.core import CoreState, callback, Context, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.config_validation import PLATFORM_SCHEMA
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from . import DOMAIN, PLATFORMS
from .controllers import SwitchController, Thermostat, AbstractController, PidParams, NumberPidController, ClimatePidController

_LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "Smart Thermostat with auto Heat/Cool modes and PID control support"
DEFAULT_PID_SAMPLE_PERIOD = "00:10:00"
DEFAULT_PID_KP = 1.0
DEFAULT_PID_KI = 1.0
DEFAULT_PID_KD = 1.0
CONF_HEATER = "heater"
CONF_COOLER = "cooler"
CONF_INVERTED = "inverted"
CONF_SENSOR = "target_sensor"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_AWAY_TEMP = "away_temp"
CONF_PRECISION = "precision"
CONF_PID_PARAMS = "pid_params"
CONF_PID_SAMPLE_PERIOD = "sample_period"
CONF_PID_MIN = "min"
CONF_PID_MAX = "max"
SUPPORT_FLAGS = SUPPORT_TARGET_TEMPERATURE
SUPPORTED_TARGET_DOMAINS = [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN, NUMBER_DOMAIN, INPUT_NUMBER_DOMAIN, CLIMATE_DOMAIN]


def _cv_pid_params_list(value: Any) -> list:
    value = cv.ensure_list_csv(value)
    if len(value) != 3:
        raise vol.Invalid(f"{CONF_PID_PARAMS} should be 3 items: kp, ki, kd")
    value = [float(v) for v in value]
    return value


def _cv_min_max_check(cfg):
    minimum = cfg.get(CONF_PID_MIN)
    maximum = cfg.get(CONF_PID_MAX)
    if None not in (minimum, maximum) and minimum >= maximum:
        raise vol.Invalid(
            f"Maximum ({minimum}) is not greater than minimum ({maximum})"
        )
    return cfg


TARGET_SCHEMA_COMMON = vol.Schema({
    vol.Required(CONF_ENTITY_ID): cv.entity_domain(SUPPORTED_TARGET_DOMAINS),
    vol.Optional(CONF_INVERTED, default=False): bool
})

TARGET_SCHEMA_SWITCH = TARGET_SCHEMA_COMMON.extend({
    vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
    vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): cv.positive_float,
    vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): cv.positive_float
})

TARGET_SCHEMA_PID_REGULATOR = vol.All(
    TARGET_SCHEMA_COMMON.extend({
        vol.Optional(CONF_PID_PARAMS,
                     default=[DEFAULT_PID_KP, DEFAULT_PID_KI, DEFAULT_PID_KD]
                     ): _cv_pid_params_list,
        vol.Optional(CONF_PID_SAMPLE_PERIOD, default=DEFAULT_PID_SAMPLE_PERIOD): cv.positive_time_period,
        vol.Optional(CONF_PID_MIN, default=None): vol.Any(None, vol.Coerce(float)),
        vol.Optional(CONF_PID_MAX, default=None): vol.Any(None, vol.Coerce(float))
    }),
    _cv_min_max_check
)

KEY_SCHEMA = vol.Schema({
    vol.Required(
        vol.Any(CONF_HEATER, CONF_COOLER),
        msg=f"Must specify at least one: '{CONF_HEATER}' or '{CONF_COOLER}'"): object
}, extra=ALLOW_EXTRA)

DATA_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_HEATER): vol.Any(
            cv.entity_domain(SUPPORTED_TARGET_DOMAINS),
            vol.Any(TARGET_SCHEMA_SWITCH, TARGET_SCHEMA_PID_REGULATOR)
        ),
        vol.Optional(CONF_COOLER): vol.Any(
            cv.entity_domain(SUPPORTED_TARGET_DOMAINS),
            vol.Any(TARGET_SCHEMA_SWITCH, TARGET_SCHEMA_PID_REGULATOR)
        ),
        vol.Required(CONF_SENSOR): cv.entity_id,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_KEEP_ALIVE): cv.positive_time_period,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_OFF]
        ),
        vol.Optional(CONF_AWAY_TEMP): vol.Coerce(float),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
    }
)

PLATFORM_SCHEMA = vol.All(KEY_SCHEMA, DATA_SCHEMA)


def _extract_target(target, schema):
    if isinstance(target, str):
        return schema({
            CONF_ENTITY_ID: target
        })
    else:
        return target


def _create_controller(name: str, mode: str, raw_conf) -> AbstractController:
    # First use common schema
    conf = _extract_target(raw_conf, TARGET_SCHEMA_COMMON)

    entity_id = conf[CONF_ENTITY_ID]
    inverted = conf[CONF_INVERTED]

    domain, _ = entity_id.split('.')

    if domain in [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN]:
        conf = _extract_target(raw_conf, TARGET_SCHEMA_SWITCH)
        min_duration = conf[CONF_MIN_DUR] if CONF_MIN_DUR in conf else None
        cold_tolerance = conf[CONF_COLD_TOLERANCE]
        hot_tolerance = conf[CONF_HOT_TOLERANCE]

        controller = SwitchController(
            name,
            mode,
            entity_id,
            cold_tolerance,
            hot_tolerance,
            inverted,
            min_duration
        )
        return controller

    elif domain in [INPUT_NUMBER_DOMAIN, NUMBER_DOMAIN]:
        conf = _extract_target(raw_conf, TARGET_SCHEMA_PID_REGULATOR)
        pid_params = conf[CONF_PID_PARAMS]

        controller = NumberPidController(
            name,
            mode,
            entity_id,
            PidParams(pid_params[0], pid_params[1], pid_params[2]),
            inverted,
            conf[CONF_PID_SAMPLE_PERIOD],
            conf[CONF_PID_MIN],
            conf[CONF_PID_MAX]
        )
        return controller

    elif domain in [CLIMATE_DOMAIN]:
        conf = _extract_target(raw_conf, TARGET_SCHEMA_PID_REGULATOR)
        pid_params = conf[CONF_PID_PARAMS]

        controller = ClimatePidController(
            name,
            mode,
            entity_id,
            PidParams(pid_params[0], pid_params[1], pid_params[2]),
            inverted,
            conf[CONF_PID_SAMPLE_PERIOD],
            conf[CONF_PID_MIN],
            conf[CONF_PID_MAX]
        )
        return controller

    else:
        raise ValueError(f"Unsupported {name} domain: '{domain}'")


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the smart thermostat platform."""

    # prevent unused variable warn
    _ = discovery_info

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name = config.get(CONF_NAME)
    sensor_entity_id = config.get(CONF_SENSOR)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)
    target_temp = config.get(CONF_TARGET_TEMP)
    keep_alive = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    away_temp = config.get(CONF_AWAY_TEMP)
    precision = config.get(CONF_PRECISION)
    unit = hass.config.units.temperature_unit
    unique_id = config.get(CONF_UNIQUE_ID)

    heater_config = config.get(CONF_HEATER)
    cooler_config = config.get(CONF_COOLER)

    cooler = _create_controller('cooler', HVAC_MODE_COOL, cooler_config) if cooler_config else None
    heater = _create_controller('heater', HVAC_MODE_HEAT, heater_config) if heater_config else None

    async_add_entities(
        [
            SmartThermostat(
                name,
                cooler,
                heater,
                sensor_entity_id,
                min_temp,
                max_temp,
                target_temp,
                keep_alive,
                initial_hvac_mode,
                away_temp,
                precision,
                unit,
                unique_id,
            )
        ]
    )


# noinspection PyAbstractClass
class SmartThermostat(ClimateEntity, RestoreEntity, Thermostat):
    """Representation of a Smart Thermostat device."""

    def __init__(
            self,
            name,
            cooler: AbstractController,
            heater: AbstractController,
            sensor_entity_id,
            min_temp,
            max_temp,
            target_temp,
            keep_alive,
            initial_hvac_mode,
            away_temp,
            precision,
            unit,
            unique_id,
    ):
        """Initialize the thermostat."""
        self._name = name
        self._cooler = cooler
        self._heater = heater
        self.sensor_entity_id = sensor_entity_id
        self._keep_alive = keep_alive
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or away_temp
        self._temp_precision = precision
        self._hvac_list = [HVAC_MODE_OFF]
        self._cur_temp = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._unit = unit
        self._unique_id = unique_id
        self._support_flags = SUPPORT_FLAGS
        if away_temp:
            self._support_flags = SUPPORT_FLAGS | SUPPORT_PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE, PRESET_AWAY]
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._away_temp = away_temp

        self._controllers = []

        if self._cooler:
            self._cooler.set_thermostat(self)
            self._controllers.append(self._cooler)
            self._hvac_list.append(HVAC_MODE_COOL)

        if self._heater:
            self._heater.set_thermostat(self)
            self._controllers.append(self._heater)
            self._hvac_list.append(HVAC_MODE_HEAT)

        if self._cooler and self._cooler:
            self._hvac_list.append(HVAC_MODE_HEAT_COOL)

    async def async_added_to_hass(self):
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_entity_id], self._async_sensor_changed
            )
        )

        for controller in self._controllers:
            await controller.async_added_to_hass(self.hass, await self.async_get_last_state())

        if self._keep_alive:
            self.async_on_remove(
                async_track_time_interval(
                    self.hass, self._async_control, self._keep_alive
                )
            )

        @callback
        def _async_startup(*_):
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
            ):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()

            _LOGGER.info("%s: Ready, supported HVAC modes: %s", self.name, self._hvac_list)

            for contr in self._controllers:
                contr.async_startup()

        if self.hass.state == CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        old_state = await self.async_get_last_state()
        if old_state is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    self._target_temp = self._get_default_target_temp()
                    _LOGGER.warning(
                        "%S: Undefined target temperature, falling back to %s",
                        self.name,
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if old_state.attributes.get(ATTR_PRESET_MODE) in self._attr_preset_modes:
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = old_state.state

        else:
            # No previous state, try and restore defaults
            self._target_temp = self._get_default_target_temp()
            _LOGGER.warning(
                "%s: No previously saved temperature, setting to %s", self.name, self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVAC_MODE_OFF

    def get_hass(self) -> HomeAssistant:
        return self.hass

    def get_entity_id(self) -> str:
        return self.entity_id

    def get_hvac_mode(self) -> str:
        return self.hvac_mode

    def get_context(self) -> Context:
        return self._context

    def get_target_temperature(self):
        return self.target_temperature

    def get_current_temperature(self):
        return self.current_temperature

    def _get_default_target_temp(self):
        return (self.max_temp + self.min_temp) / 2

    @property
    def should_poll(self):
        """Return the polling state."""
        return False

    @property
    def name(self):
        """Return the name of the thermostat."""
        return self._name

    @property
    def unique_id(self):
        """Return the unique id of this thermostat."""
        return self._unique_id

    @property
    def precision(self):
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self):
        """Return the supported step of target temperature."""
        # Since this integration does not yet have a step size parameter
        # we have to re-use the precision as the step size for now.
        return self.precision

    @property
    def temperature_unit(self):
        """Return the unit of measurement."""
        return self._unit

    @property
    def current_temperature(self):
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def hvac_mode(self):
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self):
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVAC_MODE_OFF:
            return CURRENT_HVAC_OFF
        if self._cooler and self._cooler.running and self._cooler.is_working():
            return CURRENT_HVAC_COOL
        if self._heater and self._heater.running and self._heater.is_working():
            return CURRENT_HVAC_HEAT
        return CURRENT_HVAC_IDLE

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temp

    @property
    def hvac_modes(self):
        """List of available operation modes."""
        return self._hvac_list

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        attrs = {}
        for controller in self._controllers:
            extra_controller_attrs = controller.extra_state_attributes
            if extra_controller_attrs:
                attrs = {**attrs, **extra_controller_attrs}
        return attrs

    async def async_set_hvac_mode(self, hvac_mode):
        """Set hvac mode."""

        if hvac_mode not in self._hvac_list:
            _LOGGER.error("%s: Unrecognized hvac mode: %s", self.name, hvac_mode)
            return

        self._hvac_mode = hvac_mode

        await self._async_control(force=True)

        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs):
        """Set new target temperature."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return
        self._target_temp = temperature
        await self._async_control(force=True)
        self.async_write_ha_state()

    @property
    def min_temp(self):
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self):
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    async def _async_sensor_changed(self, event):
        """Handle temperature changes."""
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(new_state)
        await self._async_control()
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state):
        """Update thermostat with latest state from sensor."""
        try:
            cur_temp = float(state.state)
            if math.isnan(cur_temp) or math.isinf(cur_temp):
                raise ValueError(f"Sensor has illegal state {state.state}")
            self._cur_temp = cur_temp
        except ValueError as ex:
            _LOGGER.error("%s: Unable to update from sensor: %s", self.name, ex)

    async def _async_control(self, time=None, force=False):
        """Call controllers"""
        async with self._temp_lock:
            cur_temp = self._cur_temp
            target_temp = self._target_temp

            need_heat = False
            need_cool = False

            debug_info = None
            _ = debug_info

            if None not in (need_cool, need_heat):
                if cur_temp == target_temp:
                    pass
                elif cur_temp > target_temp and self._hvac_mode in [HVAC_MODE_COOL, HVAC_MODE_HEAT_COOL]:
                    need_cool = True
                elif cur_temp < target_temp and self._hvac_mode in [HVAC_MODE_HEAT, HVAC_MODE_HEAT_COOL]:
                    need_heat = True
                debug_info = f"need_cool: {need_cool}, need_heat: {need_heat} (cur: {cur_temp}, target: {target_temp})"
            else:
                debug_info = f"current/target not available (cur: {cur_temp}, target: {target_temp})"

            if not need_cool and self._cooler and self._cooler.running:
                _LOGGER.debug("%s: Stopping %s, %s", self.entity_id, self._cooler.name, debug_info)
                await self._cooler.async_stop()

            if not need_heat and self._heater and self._heater.running:
                _LOGGER.debug("%s: Stopping %s, %s", self.entity_id, self._heater.name, debug_info)
                await self._heater.async_stop()

            if need_cool and (not self._cooler or not self._cooler.running):
                _LOGGER.debug("%s: Starting %s, %s", self.entity_id, self._cooler.name, debug_info)
                await self._cooler.async_start()

            if need_heat and (not self._heater or not self._heater.running):
                _LOGGER.debug("%s: Starting %s, %s", self.entity_id, self._heater.name, debug_info)
                await self._heater.async_start()

            for controller in self._controllers:
                if controller.running:
                    await controller.async_control(time=time, force=force)

    @property
    def supported_features(self):
        """Return the list of supported features."""
        return self._support_flags

    async def async_set_preset_mode(self, preset_mode: str):
        """Set new preset mode."""
        if preset_mode not in (self._attr_preset_modes or []):
            raise ValueError(
                f"Got unsupported preset_mode {preset_mode}. Must be one of {self._attr_preset_modes}"
            )
        if preset_mode == self._attr_preset_mode:
            # I don't think we need to call async_write_ha_state if we didn't change the state
            return
        if preset_mode == PRESET_AWAY:
            self._attr_preset_mode = PRESET_AWAY
            self._saved_target_temp = self._target_temp
            self._target_temp = self._away_temp
            await self._async_control(force=True)
        elif preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control(force=True)

        self.async_write_ha_state()
