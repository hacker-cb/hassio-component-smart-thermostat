"""
Adds support for smart thermostat units.
forked from HA-core `generic_thermostat` 827501659c926ace3741425760b1294d2e93b48e
"""
import asyncio
import logging
import math
from typing import Mapping, Any, Optional, List

import voluptuous as vol
from voluptuous import ALLOW_EXTRA

from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate import DOMAIN as CLIMATE_DOMAIN
from homeassistant.components.climate.const import (
    ATTR_PRESET_MODE,
    CURRENT_HVAC_COOL,
    CURRENT_HVAC_HEAT,
    CURRENT_HVAC_IDLE,
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
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
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
from homeassistant.core import CoreState, callback, Context, HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.config_validation import PLATFORM_SCHEMA
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from . import DOMAIN, PLATFORMS
from .controllers import SwitchController, Thermostat, AbstractController, PidParams, NumberPidController, ClimatePidController, REASON_THERMOSTAT_FIRST_RUN, \
    REASON_THERMOSTAT_HVAC_MODE_CHANGED, REASON_THERMOSTAT_TARGET_TEMP_CHANGED, REASON_THERMOSTAT_SENSOR_CHANGED, REASON_CONTROL_ENTITY_CHANGED, \
    PwmSwitchPidController

_LOGGER = logging.getLogger(__name__)

DEFAULT_SWITCH_TOLERANCE = 0.3
DEFAULT_HEAT_COOL_TOLERANCE = 0.3
DEFAULT_NAME = "Smart Thermostat"
# DEFAULT_PID_KP = 1.0
# DEFAULT_PID_KI = 1.0
# DEFAULT_PID_KD = 1.0
CONF_HEATER = "heater"
CONF_COOLER = "cooler"
CONF_INVERTED = "inverted"
CONF_SENSOR = "target_sensor"
CONF_STALE_DURATION = "sensor_stale_duration"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_HEAT_COOL_COLD_TOLERANCE = "heat_cool_cold_tolerance"
CONF_HEAT_COOL_HOT_TOLERANCE = "heat_cool_hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_AWAY_TEMP = "away_temp"
CONF_HEAT_COOL_DISABLED = "heat_cool_disabled"
CONF_PRECISION = "precision"
CONF_PID_PARAMS = "pid_params"
CONF_PID_SAMPLE_PERIOD = "pid_sample_period"
CONF_PID_MIN = "min"
CONF_PID_MAX = "max"
CONF_PID_SWITCH_ENTITY_ID = "switch_entity_id"
CONF_PID_SWITCH_INVERTED = "switch_inverted"
CONF_PWM_SWITCH_PERIOD = "pwm_period"

SUPPORT_FLAGS = SUPPORT_TARGET_TEMPERATURE
SUPPORTED_TARGET_DOMAINS = [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN, NUMBER_DOMAIN, INPUT_NUMBER_DOMAIN, CLIMATE_DOMAIN]


#
ATTR_LAST_ASYNC_CONTROL_HVAC_MODE = "async_control_hvac_mode"


def _cv_pid_params_list(value: Any) -> list:
    value = cv.ensure_list_csv(value)
    if len(value) != 3:
        raise vol.Invalid(f"{CONF_PID_PARAMS} should be 3 items: kp, ki, kd")
    return [cv.positive_float(v) for v in value]


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
    vol.Optional(CONF_INVERTED, default=False): bool,
    vol.Optional(CONF_KEEP_ALIVE, default=None): vol.Any(None, cv.positive_time_period)
})

TARGET_SCHEMA_SWITCH = TARGET_SCHEMA_COMMON.extend({
    vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
    vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_SWITCH_TOLERANCE): cv.positive_float,
    vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_SWITCH_TOLERANCE): cv.positive_float
})

TARGET_SCHEMA_PID_REGULATOR_COMMON = TARGET_SCHEMA_COMMON.extend({
    vol.Required(CONF_PID_PARAMS
                 # ,default=[DEFAULT_PID_KP, DEFAULT_PID_KI, DEFAULT_PID_KD]
                 ): _cv_pid_params_list,
    vol.Optional(CONF_PID_SAMPLE_PERIOD, default=None): vol.Any(None, cv.positive_time_period),
    vol.Optional(CONF_PID_MIN, default=None): vol.Any(None, vol.Coerce(float)),
    vol.Optional(CONF_PID_MAX, default=None): vol.Any(None, vol.Coerce(float))
})

TARGET_SCHEMA_PID_REGULATOR_PWM_SWITCH = TARGET_SCHEMA_PID_REGULATOR_COMMON.extend({
    vol.Required(CONF_PID_PARAMS): _cv_pid_params_list,
    vol.Optional(CONF_PID_SAMPLE_PERIOD, default=None): vol.Any(None, cv.positive_time_period),
    vol.Required(CONF_PWM_SWITCH_PERIOD): cv.positive_time_period
})

TARGET_SCHEMA_PID_REGULATOR_CLIMATE = TARGET_SCHEMA_PID_REGULATOR_COMMON.extend({
})

TARGET_SCHEMA_PID_REGULATOR_NUMBER = TARGET_SCHEMA_PID_REGULATOR_COMMON.extend({
    vol.Required(CONF_PID_SWITCH_ENTITY_ID): cv.entity_domain([SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN]),
    vol.Optional(CONF_PID_SWITCH_INVERTED, default=False): vol.Coerce(bool)
})


def _cv_controller_target(cfg: Any) -> Any:
    entity_id: str

    if isinstance(cfg, str):
        entity_id = cfg
        cfg = {
            CONF_ENTITY_ID: entity_id
        }

    if CONF_ENTITY_ID not in cfg:
        raise vol.Invalid(f"{CONF_ENTITY_ID} should be specified")

    entity_id = cfg[CONF_ENTITY_ID]

    domain = split_entity_id(entity_id)[0]

    cfg = _cv_min_max_check(cfg)

    if domain in [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN]:
        if CONF_PID_PARAMS in cfg:
            return TARGET_SCHEMA_PID_REGULATOR_PWM_SWITCH(cfg)
        else:
            return TARGET_SCHEMA_SWITCH(cfg)
    elif domain in [INPUT_NUMBER_DOMAIN, NUMBER_DOMAIN]:
        return TARGET_SCHEMA_PID_REGULATOR_NUMBER(cfg)
    elif domain in [CLIMATE_DOMAIN]:
        return TARGET_SCHEMA_PID_REGULATOR_CLIMATE(cfg)
    else:
        raise vol.Invalid(f"{entity_id}:  Unsupported domain: {domain}")


KEY_SCHEMA = vol.Schema({
    vol.Required(
        vol.Any(CONF_HEATER, CONF_COOLER),
        msg=f"Must specify at least one: '{CONF_HEATER}' or '{CONF_COOLER}'"): object
}, extra=ALLOW_EXTRA)

DATA_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Optional(CONF_HEATER): vol.Any(_cv_controller_target, [_cv_controller_target]),
        vol.Optional(CONF_COOLER): vol.Any(_cv_controller_target, [_cv_controller_target]),
        vol.Required(CONF_SENSOR): cv.entity_id,
        vol.Optional(CONF_STALE_DURATION): vol.All(
            cv.time_period, cv.positive_timedelta
        ),
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_AWAY_TEMP): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_HEAT_COOL_DISABLED): vol.Coerce(bool),
        vol.Optional(CONF_HEAT_COOL_COLD_TOLERANCE, default=DEFAULT_HEAT_COOL_TOLERANCE): cv.positive_float,
        vol.Optional(CONF_HEAT_COOL_HOT_TOLERANCE, default=DEFAULT_HEAT_COOL_TOLERANCE): cv.positive_float,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        )
    }
)

PLATFORM_SCHEMA = vol.All(KEY_SCHEMA, DATA_SCHEMA)


def _create_controllers(
        prefix: str,
        mode: str,
        conf_list: Any,
        heat_cool_cold_tolerance: float,
        heat_cool_hot_tolerance: float
) -> [AbstractController]:
    if conf_list is None:
        return []
    if not isinstance(conf_list, list):
        conf_list = [conf_list]

    controllers: List[AbstractController] = []

    controller_number = 1

    for conf in conf_list:
        name = f"{prefix}_{controller_number}"

        entity_id = conf[CONF_ENTITY_ID]
        inverted = conf[CONF_INVERTED]
        keep_alive = conf[CONF_KEEP_ALIVE]

        domain = split_entity_id(entity_id)[0]

        controller = None

        if domain in [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN]:
            if CONF_PID_PARAMS in conf:
                pid_params = conf[CONF_PID_PARAMS]

                controller = PwmSwitchPidController(
                    name,
                    mode,
                    entity_id,
                    PidParams(pid_params[0], pid_params[1], pid_params[2]),
                    conf[CONF_PID_SAMPLE_PERIOD],
                    inverted,
                    keep_alive,
                    conf[CONF_PWM_SWITCH_PERIOD]
                )
            else:
                min_duration = conf[CONF_MIN_DUR] if CONF_MIN_DUR in conf else None
                cold_tolerance = conf[CONF_COLD_TOLERANCE]
                hot_tolerance = conf[CONF_HOT_TOLERANCE]

                if cold_tolerance < heat_cool_cold_tolerance:
                    _LOGGER.warning(
                        "cold_tolerance (%s) < heat_cool_cold_tolerance (%s). "
                        "%s will be enabled in heat/cool mode based on heat_cool_cold_tolerance (entity_id: %s).",
                        cold_tolerance,
                        heat_cool_cold_tolerance,
                        name,
                        entity_id
                    )

                if hot_tolerance < heat_cool_hot_tolerance:
                    _LOGGER.warning(
                        "hot_tolerance (%s) < heat_cool_hot_tolerance (%s). "
                        "%s will be enabled in heat/cool mode based on heat_cool_hot_tolerance (entity_id: %s).",
                        hot_tolerance,
                        heat_cool_hot_tolerance,
                        name,
                        entity_id
                    )

                controller = SwitchController(
                    name,
                    mode,
                    entity_id,
                    cold_tolerance,
                    hot_tolerance,
                    inverted,
                    keep_alive,
                    min_duration
                )

        elif domain in [INPUT_NUMBER_DOMAIN, NUMBER_DOMAIN]:
            pid_params = conf[CONF_PID_PARAMS]

            controller = NumberPidController(
                name,
                mode,
                entity_id,
                PidParams(pid_params[0], pid_params[1], pid_params[2]),
                conf[CONF_PID_SAMPLE_PERIOD],
                inverted,
                keep_alive,
                conf[CONF_PID_MIN],
                conf[CONF_PID_MAX],
                conf[CONF_PID_SWITCH_ENTITY_ID],
                conf[CONF_PID_SWITCH_INVERTED]
            )

        elif domain in [CLIMATE_DOMAIN]:
            pid_params = conf[CONF_PID_PARAMS]

            controller = ClimatePidController(
                name,
                mode,
                entity_id,
                PidParams(pid_params[0], pid_params[1], pid_params[2]),
                conf[CONF_PID_SAMPLE_PERIOD],
                inverted,
                keep_alive,
                conf[CONF_PID_MIN],
                conf[CONF_PID_MAX]
            )

        else:
            _LOGGER.error(f"Unsupported {name} domain: '{domain}' for entity {entity_id}")

        if controller:
            controllers.append(controller)
            controller_number += 1

    return controllers


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the smart thermostat platform."""

    # prevent unused variable warn
    _ = discovery_info

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name = config.get(CONF_NAME)
    sensor_entity_id = config.get(CONF_SENSOR)
    sensor_stale_duration = config.get(CONF_STALE_DURATION)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)
    target_temp = config.get(CONF_TARGET_TEMP)
    heat_cool_disabled = config.get(CONF_HEAT_COOL_DISABLED)
    heat_cool_cold_tolerance = config.get(CONF_HEAT_COOL_COLD_TOLERANCE)
    heat_cool_hot_tolerance = config.get(CONF_HEAT_COOL_HOT_TOLERANCE)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    away_temp = config.get(CONF_AWAY_TEMP)
    precision = config.get(CONF_PRECISION)
    unit = hass.config.units.temperature_unit
    unique_id = config.get(CONF_UNIQUE_ID)

    heater_config = config.get(CONF_HEATER)
    cooler_config = config.get(CONF_COOLER)

    controllers = []

    if cooler_config:
        controllers += _create_controllers(
            'cooler',
            HVAC_MODE_COOL,
            cooler_config,
            heat_cool_cold_tolerance,
            heat_cool_hot_tolerance
        )

    if heater_config:
        controllers += _create_controllers(
            'heater',
            HVAC_MODE_HEAT,
            heater_config,
            heat_cool_cold_tolerance,
            heat_cool_hot_tolerance
        )

    async_add_entities(
        [
            SmartThermostat(
                name,
                controllers,
                sensor_entity_id,
                sensor_stale_duration,
                min_temp,
                max_temp,
                target_temp,
                heat_cool_disabled,
                heat_cool_cold_tolerance,
                heat_cool_hot_tolerance,
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
            controllers: [AbstractController],
            sensor_entity_id,
            sensor_stale_duration,
            min_temp,
            max_temp,
            target_temp,
            heat_cool_disabled,
            heat_cool_cold_tolerance,
            heat_cool_hot_tolerance,
            initial_hvac_mode,
            away_temp,
            precision,
            unit,
            unique_id,
    ):
        """Initialize the thermostat."""
        self._name = name
        self._controllers = controllers
        self.sensor_entity_id = sensor_entity_id
        self._hvac_mode = initial_hvac_mode
        self._last_async_control_hvac_mode = None
        self._saved_target_temp = target_temp or away_temp
        self._temp_precision = precision
        self._hvac_list = [HVAC_MODE_OFF]
        self._cur_temp = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._heat_cool_cold_tolerance = heat_cool_cold_tolerance
        self._heat_cool_hot_tolerance = heat_cool_hot_tolerance
        self._unit = unit
        self._unique_id = unique_id
        self._support_flags = SUPPORT_FLAGS
        self._hvac_action = CURRENT_HVAC_IDLE
        self._sensor_stale_duration = sensor_stale_duration
        self._remove_stale_tracking = None
        if away_temp:
            self._support_flags = SUPPORT_FLAGS | SUPPORT_PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE, PRESET_AWAY]
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._away_temp = away_temp

        for controller in self._controllers:
            controller.set_thermostat(self)
            if controller.mode == HVAC_MODE_HEAT and HVAC_MODE_HEAT not in self._hvac_list:
                self._hvac_list.append(HVAC_MODE_HEAT)
            elif controller.mode == HVAC_MODE_COOL and HVAC_MODE_COOL not in self._hvac_list:
                self._hvac_list.append(HVAC_MODE_COOL)

        if (HVAC_MODE_COOL in self._hvac_list and HVAC_MODE_HEAT in self._hvac_list) and not heat_cool_disabled:
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

        old_state = await self.async_get_last_state()

        for controller in self._controllers:
            attrs = old_state.attributes.get(controller.get_unique_id(), {}) if old_state else  {}

            await controller.async_added_to_hass(self.hass, attrs)

            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, controller.get_used_entity_ids(), self._async_controller_target_entity_changed
                )
            )

        async def _async_first_run():
            """Will called one time. Need on hot reload when HA core is running"""
            await self._async_control(reason=REASON_THERMOSTAT_FIRST_RUN)
            self.async_write_ha_state()

        @callback
        async def _async_startup(*_):
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
            ):
                await self._async_update_temp(sensor_state.state)
                self.async_write_ha_state()

            _LOGGER.info("%s: Ready, supported HVAC modes: %s, Stale duration: %s",
                         self.name, self._hvac_list, self._sensor_stale_duration)

            self.hass.create_task(_async_first_run())

        if self.hass.state == CoreState.running:
            await _async_startup()
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
            self._last_async_control_hvac_mode = old_state.attributes.get(ATTR_LAST_ASYNC_CONTROL_HVAC_MODE)

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

        action = CURRENT_HVAC_IDLE

        if self._hvac_action == CURRENT_HVAC_IDLE:
            pass

        elif self._hvac_action == CURRENT_HVAC_COOL:
            for controller in self._controllers:
                if controller.mode == HVAC_MODE_COOL and controller.working:
                    action = CURRENT_HVAC_COOL

        elif self._hvac_action == CURRENT_HVAC_HEAT:
            for controller in self._controllers:
                if controller.mode == HVAC_MODE_HEAT and controller.working:
                    action = CURRENT_HVAC_HEAT

        return action

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
                attrs[controller.get_unique_id()] = extra_controller_attrs
        attrs[ATTR_LAST_ASYNC_CONTROL_HVAC_MODE] = self._last_async_control_hvac_mode
        return attrs

    async def async_set_hvac_mode(self, hvac_mode):
        """Set hvac mode."""

        if hvac_mode not in self._hvac_list:
            _LOGGER.error("%s: Unrecognized hvac mode: %s", self.name, hvac_mode)
            return

        self._hvac_mode = hvac_mode

        await self._async_control(force=True, reason=REASON_THERMOSTAT_HVAC_MODE_CHANGED)

        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs):
        """Set new target temperature."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return
        self._target_temp = temperature
        await self._async_control(force=True, reason=REASON_THERMOSTAT_TARGET_TEMP_CHANGED)
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
        if new_state is None:
            return

        await self._async_update_temp(new_state.state)

        if self._cur_temp is not None and self._sensor_stale_duration:
            if self._remove_stale_tracking:
                self._remove_stale_tracking()
            self._remove_stale_tracking = async_track_time_interval(
                self.hass,
                self._async_sensor_not_responding,
                self._sensor_stale_duration,
            )

        await self._async_control(reason=REASON_THERMOSTAT_SENSOR_CHANGED)
        self.async_write_ha_state()

    @callback
    async def _async_sensor_not_responding(self, now=None):
        """Handle sensor stale event."""

        _LOGGER.debug(
            "%s: Sensor has not been updated for %s",
            self.entity_id,
            now - self.hass.states.get(self.sensor_entity_id).last_updated,
        )
        _LOGGER.warning("%s: Sensor is stalled, all controllers will be stopped.", self.entity_id)
        await self._async_update_temp(None)

    async def _async_controller_target_entity_changed(self, event):
        """Handle controller target entity changes."""
        _ = event
        await self._async_control(reason=REASON_CONTROL_ENTITY_CHANGED)
        self.async_write_ha_state()

    @callback
    async def _async_update_temp(self, temp):
        """Update thermostat with latest state from sensor."""
        if temp in (STATE_UNAVAILABLE, STATE_UNKNOWN, None):
            self._cur_temp = None
            return

        try:
            self._cur_temp = float(temp)
            if math.isnan(self._cur_temp) or math.isinf(self._cur_temp):
                raise ValueError(f"Sensor has illegal value {temp}")

        except (ValueError, TypeError) as ex:
            self._cur_temp = None
            _LOGGER.error("%s: Unable to update from sensor: %s", self.name, ex)

    async def _async_control(self, time=None, force=False, reason=None):
        """Call controllers"""
        async with self._temp_lock:
            cur_temp = self._cur_temp
            target_temp = self._target_temp

            debug_info = None
            _ = debug_info

            new_hvac_action = self._hvac_action

            if self._last_async_control_hvac_mode != self._hvac_mode:
                _LOGGER.info("%s: mode changed: %s -> %s", self.entity_id, self._last_async_control_hvac_mode, self._hvac_mode)
            elif self._hvac_mode == HVAC_MODE_OFF:
                # Skip control, last `OFF` was already processed
                return

            if self._hvac_mode == HVAC_MODE_OFF:
                new_hvac_action = CURRENT_HVAC_IDLE

            if None not in (cur_temp, target_temp):

                too_cold = cur_temp <= target_temp - self._heat_cool_cold_tolerance
                too_hot = cur_temp >= target_temp + self._heat_cool_hot_tolerance

                if self._hvac_mode == HVAC_MODE_COOL or (self._hvac_mode == HVAC_MODE_HEAT_COOL and too_hot):
                    new_hvac_action = CURRENT_HVAC_COOL
                elif self._hvac_mode == HVAC_MODE_HEAT or (self._hvac_mode == HVAC_MODE_HEAT_COOL and too_cold):
                    new_hvac_action = CURRENT_HVAC_HEAT
                debug_info = f"hvac_action: {new_hvac_action}, (cur: {cur_temp}, target: {target_temp})"
            else:
                new_hvac_action = CURRENT_HVAC_IDLE
                debug_info = f"current/target not available (cur: {cur_temp}, target: {target_temp})"

            # _LOGGER.debug("%s: HVAC old: %s, new: %s (%s): %s", self.entity_id, self._hvac_action, new_hvac_action, reason, debug_info)

            if self._hvac_action != new_hvac_action:
                _LOGGER.info("%s: Changed controllers HVAC action from %s to %s (cur: %s, target: %s, mode: %s)",
                             self.entity_id, self._hvac_action, new_hvac_action, cur_temp, target_temp, self._hvac_mode)
                self._hvac_action = new_hvac_action

            # Stop all controllers which are not needed
            for controller in self._controllers:
                controller_debug_info = f"{debug_info}, running: {controller.running}, working: {controller.working}"

                # _LOGGER.debug("%s: Check for stop %s, %s", self.entity_id, controller.name, controller_debug_info)

                if (
                        self._hvac_action != CURRENT_HVAC_COOL and
                        controller.mode == HVAC_MODE_COOL and
                        (controller.running or (controller.working and self._hvac_mode != HVAC_MODE_OFF))
                ):
                    _LOGGER.debug("%s: Stopping %s, %s", self.entity_id, controller.name, controller_debug_info)
                    await controller.async_stop()
                if (
                        self._hvac_action != CURRENT_HVAC_HEAT and
                        controller.mode == HVAC_MODE_HEAT and
                        (controller.running or (controller.working and self._hvac_mode != HVAC_MODE_OFF))
                ):
                    _LOGGER.debug("%s: Stopping %s, %s", self.entity_id, controller.name, controller_debug_info)
                    await controller.async_stop()

            # Start all controllers which are needed
            for controller in self._controllers:
                controller_debug_info = f"{debug_info}, running: {controller.running}, working: {controller.working}"

                # _LOGGER.debug("%s: Check for start %s, %s", self.entity_id, controller.name, controller_debug_info)

                if (
                        self._hvac_action == CURRENT_HVAC_COOL and
                        controller.mode == HVAC_MODE_COOL and
                        not controller.running
                ):
                    _LOGGER.debug("%s: Starting %s, %s", self.entity_id, controller.name, controller_debug_info)
                    await controller.async_start()
                if (
                        self._hvac_action == CURRENT_HVAC_HEAT and
                        controller.mode == HVAC_MODE_HEAT and
                        not controller.running
                ):
                    _LOGGER.debug("%s: Starting %s, %s", self.entity_id, controller.name, controller_debug_info)
                    await controller.async_start()

            # Call async_control() on running controllers
            for controller in self._controllers:
                await controller.async_control(time=time, force=force, reason=reason)

            self._last_async_control_hvac_mode = self._hvac_mode

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
        reason = "preset_changed"
        if preset_mode == PRESET_AWAY:
            self._attr_preset_mode = PRESET_AWAY
            self._saved_target_temp = self._target_temp
            self._target_temp = self._away_temp
            await self._async_control(force=True, reason=reason)
        elif preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control(force=True, reason=reason)

        self.async_write_ha_state()
