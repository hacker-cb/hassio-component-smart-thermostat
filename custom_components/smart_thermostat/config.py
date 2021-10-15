import voluptuous as vol
from voluptuous import ALLOW_EXTRA

from homeassistant.components.climate import SUPPORT_TARGET_TEMPERATURE, HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_OFF
from homeassistant.components.input_boolean import DOMAIN as INPUT_BOOLEAN_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.const import CONF_ENTITY_ID, CONF_NAME, PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE, CONF_UNIQUE_ID
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.config_validation import PLATFORM_SCHEMA

DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "Smart Thermostat with auto Heat/Cool modes and PID control support"
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
SUPPORT_FLAGS = SUPPORT_TARGET_TEMPERATURE
SUPPORTED_TARGET_DOMAINS = [SWITCH_DOMAIN, INPUT_BOOLEAN_DOMAIN]

TARGET_SCHEMA = vol.Schema({
    vol.Required(CONF_ENTITY_ID): cv.entity_domain(SUPPORTED_TARGET_DOMAINS),
    vol.Optional(CONF_INVERTED, default=False): bool,
    vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
    vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
    vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float)
})

KEY_SCHEMA = vol.Schema({
    vol.Required(
        vol.Any(CONF_HEATER, CONF_COOLER),
        msg=f"Must specify at least one: '{CONF_HEATER}' or '{CONF_COOLER}'"): object
}, extra=ALLOW_EXTRA)

DATA_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HEATER): vol.Any(cv.entity_domain(SUPPORTED_TARGET_DOMAINS), TARGET_SCHEMA),
        vol.Required(CONF_COOLER): vol.Any(cv.entity_domain(SUPPORTED_TARGET_DOMAINS), TARGET_SCHEMA),
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
