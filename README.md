# Smart Thermostat with auto Heat/Cool mode and PID control support (Home Assistant component)

Thermostat was designed to control different kind of entities (not only switchable like in [generic_thermostat]).

### Supported domains and modes for heaters and coolers:

* `switch`, `input_boolean` - Basic on/off like [generic_thermostat] or PWM mode.
* `climate` - PID regulator.
* `number`, `input_number` -  PID regulator (additional switch is required).

### Current features:

* Support multiply heaters/coolers.
* Supports `heat_cool` mode.
* Supports `away` mode.
* Supports invert logic of the heater/cooler.
* Protection if target sensor does not report values for period of time (`sensor_stale_duration`).

## Installation (Manual)

1. Copy `/custom_components/smart_thermostat` to your `<config_dir>/custom_components/` directory.

   * On HassIO the final location will be `/config/custom_components/smart_thermostat`.
   * On Supervised the final location will be `/usr/share/hassio/homeassistant/custom_components/smart_thermostat`.
   * _NOTE: You will need to create the `custom_components` folder if it does not exist._

1. Restart Home Assistant Core.

## Minimal working config example
```yaml
climate:
  - platform: smart_thermostat    
    name: kitchen_thermostat
    target_sensor: sensor.kitchen_temperature
    min_temp: 15
    max_temp: 28
    precision: 0.1    
    heater:
      - entity_id: climate.kitchen_thermostat_heating_floor
        pid_params: 1.3, 0.5, 0.2
      - entity_id: input_number.kitchen_custom_ajustable_heater_regulator
        switch_entity_id: input_boolean.kitchen_custom_ajustable_heater_regulator_switch
        pid_params: 1.3, 0.5, 0.2
    cooler: switch.kitchen_on_off_cooler
```

## Glossary

* `target_temp` - Climate target temperature, can be changed in UI. Initial can be set via `target_temp` config option.
* `cur_temp` - Current sensor temperature. Will be reported by `target_sensor` entity.
* `CONFIG.xxx` - Reference to the config option.
* `CONFIG.CONTROLLER.xxx` - Reference to the config controller option (**heater/cooler**).


## Common config options

* `name` _(Required)_ - Climate entity name
* `unique_id` _(Optional)_ - Climate entity `unique_id`
* `cooler` _(Optional)_ - String, Array or Map of the coolers.
* `heater` _(Optional)_ - String, Array or Map of the heaters.
* `target_sensor` _(Required)_ - Target temperature sensor
* `sensor_stale_duration` _(Optional)_ - Thermostat will stop all controllers if no data received from sensor during this period.
* `min_temp` _(Optional, default=7)_ - Set minimum set point available.
* `max_temp` _(Optional, default=35)_ - Set maximum set point available.
* `away_temp` _(Optional)_ - Temperature used by the `away` mode. If this is not specified, the preset mode feature will not be available.
* `target_temp` _(Optional)_ - Initial target temperature.
* `heat_cool_cold_tolerance` _(Optional, default=0.3)_ - Cold tolerance for turning on heater controllers. Used only in `heat_cool` mode.
* `heat_cool_hot_tolerance` _(Optional, default=0.3)_ - Hot tolerance for turning on cooler controllers. Used only in `heat_cool` mode.
* `initial_hvac_mode` _(Optional)_ - Initial HVAC mode.
* `precision` _(Optional)_ - Precision for this device. Supported values are 0.1, 0.5 and 1.0. Default: 0.1 for Celsius and 1.0 for Fahrenheit.

_NOTE: at least one of `heater` or `cooler` is required._

## Common behavior

Initial HVAC mode can be set via `initial_hvac_mode` config option.

Thermostat behavior will depend on active HVAC mode. HVAC mode can be set in UI. 

_**NOTE: Smart thermostat will always take full control of the heaters/coolers. 
So it will turn them on/off back if you change their states manually**_

### HVAC_MODE = `heat`
_NOTE: available if at least one `CONFIG.heater` was defined._

* All **heater** controllers will be turned on. Specific behavior of each **heater** will depend on the controller type.
* All **cooler** controllers will be turned off.

### HVAC_MODE = `cool`

_NOTE: available if at least one `CONFIG.coller` was defined._

* All **cooler** controllers will be turned on. Specific behavior of each **cooler** will depend on the controller type.
* All **heater** controllers will be turned off.

### HVAC_MODE = `heat_cool` 

_NOTE: available if at least one `CONFIG.heater` and at least one `CONFIG.cooler` were defined._

* If `cur_temp >= target_temp + CONFIG.heat_cool_hot_tolerance` 
  * All **cooler** controllers will be turned on.
  * All **heater** controllers will be turned off.
  * Specific behavior of each heater/cooler will depend on the controller type.


* If `cur_temp <= target_temp - CONFIG.heat_cool_cold_tolerance` 
  * All **heater** controllers will be turned on.
  * All **cooler** controllers will be turned off.
  * Specific behavior of each heater/cooler will depend on the controller type.

_NOTE: turning on controller **DOES NOT MEANS** turning on `CONFIG.CONTROLLER.enitity_id` inside controller. 
Controller behavior depends on the **specific controller logic** and described below for each controller._


## Controllers

Specific controller will be created for each `heater`/`cooler` config option based on `CONFIG.CONTROLLER.enitity_id` domain. 

### Switch controller (ON/OFF)

Domains: `switch`,`input_boolean` 

#### Config options

* `entity_id` _(Required)_ - Target entity ID.
* `inverted` _(Optional, default=false)_ - Need to invert `entity_id` logic.
* `keep_alive` _(Optional)_ - Send keep-alive interval. Use with heaters, coolers,  A/C units that shut off if they don’t receive a signal from their remote for a while. 
* `min_cycle_duration` _(Optional, default=null)_ - Minimal cycle duration. Used to protect from on/off cycling.
* `cold_tolerance` _(Optional, default=0.3)_ - Cold tolerance.
* `hot_tolerance` _(Optional, default=0.3)_ - Hot tolerance.

#### Behavior

* Turn on `entity_id` if `cur_temp <= target_temp - cold_tolerance` (heater) or `cur_temp >= target_temp + self._hot_tolerance` (cooler)
* No `entity_id` changes will be performed if config `min_cycle_duration` was set and enough time was not passed since last switch.
* Behavior on/off will be inverted if `inverted` config option was set to `true`

### PWM Switch PID controller

Domains: `switch`,`input_boolean`.

See [General PID explanation](docs/PID.md) how to tune PID parameters.

* Internal PID limits are integers, defined as constants `PWM_SWITCH_MIN_VALUE` and `PWM_SWITCH_MAX_VALUE` (0, 100).
  So, you must use this limits when tuning `pid_params` terms. 

#### Config options

* `entity_id` _(Required)_ - Target entity ID.
* `inverted` _(Optional, default=false)_ - Need to invert `entity_id` logic.
* `keep_alive` _(Optional)_ - Send keep-alive interval. Use with heaters, coolers,  A/C units that shut off if they don’t receive a signal from their remote for a while. 
* `pid_params` _(Required)_ - PID params comma-separated string or array in the format `Kp, Ki, Kd` (_Always positive, will be inverted internally for cool mode_).
* `pid_sample_period` _(Optional)_ - PID constant sample time period.
* `pwm_period`  _(Required)_ - PWM period. Switch will be turned on and turned off according internal PID output once in this period.  

#### Behavior

* PID output will be calculated internally based on provided `pid_params`.
* `pwm_period` will be separated to two parts: `ON` and `OFF`. Each part duration will depend on PID output. 
* PWM on/off need will be checked every `pwm_period/100` time **but not often than each 1 second**. (`PWM_SWITCH_MAX_VALUE` internal const variable)
* Behavior on/off will be inverted if `inverted` config option was set to `true`.
* It is keep on/off state duration before Home Assistant restart. Last change time is saved in thermostat state attributes.

NOTE: This mode will be set if entity domain is one of the listed above and `pid_params` config entry is present.

### Climate controller (PID mode supported)

Domains: `climate`

See [General PID explanation](docs/PID.md) how to tune PID parameters.

#### Config options

* `entity_id` _(Required)_ - Target entity ID.
* `inverted` _(Optional, default=false)_ - Need to invert `entity_id` logic.
* `keep_alive` _(Optional)_ - Send keep-alive interval. Use with heaters, coolers,  A/C units that shut off if they don’t receive a signal from their remote for a while. 
* `pid_params` _(Required)_ - PID params comma-separated string or array in the format `Kp, Ki, Kd` (_Always positive, will be inverted internally for cool mode_).
* `pid_sample_period` _(Optional)_ - PID constant sample time period.
* `min` _(Optional)_ - Minimum temperature which can be set. Attribute `min_temp` from `entity_id` will be used if not specified.
* `max` _(Optional)_ - Maximum temperature which can be set. Attribute `max_temp` from `entity_id` will be used if not specified.

#### Behavior

* Climate `entity_id` will be turned on when controller is active.
* Climate `entity_id` will be turned off when controller is not active.
* Climate `entity_id` temperature will be adjusted every `pid_sample_period` it is provided, or on every `CONFIF.target_sensor` update if `pid_sample_period` is not provided.
* `pid_params` will be inverted if `inverted` was set to `true`

### Number + Switch controller (PID mode supported)

Domains: `number`,`input_number`

See [General PID explanation](docs/PID.md) how to tune PID parameters.

#### Config options

* `entity_id` _(Required)_ - Target entity ID.
* `inverted` _(Optional, default=false)_ - Need to invert `entity_id` logic.
* `keep_alive` _(Optional)_ - Send keep-alive interval. Use with heaters, coolers,  A/C units that shut off if they don’t receive a signal from their remote for a while. 
* `pid_params` _(Required)_ - PID params comma-separated string or array in the format `Kp, Ki, Kd` (_Always positive, will be inverted internally for cool mode_).
* `pid_sample_period` _(Optional)_ - PID constant sample time period.
* `min` _(Optional)_ - Minimum temperature which can be set. Attribute `min` from `entity_id` will be used if not specified.
* `max` _(Optional)_ - Maximum temperature which can be set. Attribute `max` from `entity_id` will be used if not specified.
* `switch_entity_id` _(Required)_ - Switch entity which belongs to `switch`,`input_boolean` domains.
* `switch_inverted` _(Optional, default=false)_ - Is `switch_entity_id` inverted?

#### Behavior

* Switch `switch_entity_id` will be turned on when controller is active.
* Switch `switch_entity_id` will be turned off when controller is not active.
* Number `entity_id` temperature will be adjusted every `pid_sample_period` it is provided, or on every `CONFIF.target_sensor` update if `pid_sample_period` is not provided.
* `pid_params` will be inverted if `inverted` was set to `true`
* `switch_entity_id` behavior will be inverted if `switch_inverted` was set to `true`



## Reporting an Issue

1. Setup your logger to print debug messages for this component using:
```yaml
logger:
  default: info
  logs:
    custom_components.smart_thermostat: debug
```
1. Restart HA
1. Verify you're still having the issue
1. File an issue in this GitHub Repository containing your HA log (Developer section > Info > Load Full Home Assistant Log)
   * You can paste your log file at pastebin https://pastebin.com/ and submit a link.
   * Please include details about your setup (Pi, NUC, etc, docker?, HassOS?)
   * The log file can also be found at `/<config_dir>/home-assistant.log`
   

[generic_thermostat]: https://www.home-assistant.io/integrations/generic_thermostat/]
[General PID explanation](docs/PID.md)