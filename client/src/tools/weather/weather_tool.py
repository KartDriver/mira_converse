#!/usr/bin/env python3
"""Tool for getting weather information using Open-Meteo API."""

import json
import logging
import asyncio
import aiohttp
from typing import List, Tuple
from ..base_tool import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "weather"
        self.description = "Get weather information for a location"
        self.args = ["city", "state", "country"]
        self.llm_response = False  # Weather tool returns formatted text directly
        self.needs_translation = True  # Enable translation for non-English languages
    
    async def _get_coordinates(self, city: str, state: str = None, country: str = None) -> Tuple[float, float, dict]:
        """Get coordinates for a location using Open-Meteo Geocoding API.
        
        Returns:
            Tuple containing latitude, longitude, and location details dictionary
        """
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = "https://geocoding-api.open-meteo.com/v1/search"
                params = {
                    "name": city,
                    "count": 10,
                    "language": "en",
                    "format": "json"
                }
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise ValueError("Unable to search for location. Please try again later.")
                    
                    data = await response.json()
                    if not data.get("results"):
                        raise ValueError(f"Could not find location: {city}")
                    
                    results = data["results"]
                    original_results = results.copy()
                    
                    # Prioritize country matches if country is provided, but don't require them
                    if country:
                        country_results = [r for r in results if r.get("country_code") == country.upper()]
                        if country_results:
                            logger.debug(f"Found {len(country_results)} results matching country {country}")
                            results = country_results
                        else:
                            logger.debug(f"No results found for country {country}, using best city match instead")
                    
                    # Prioritize state matches if state is provided, but don't require them
                    if state:
                        state_results = [r for r in results if r.get("admin1", "").upper() == state.upper()]
                        if state_results:
                            logger.debug(f"Found {len(state_results)} results matching state {state}")
                            results = state_results
                        else:
                            logger.debug(f"No results found for state {state}, using best available match instead")
                    
                    # If we filtered too aggressively and have no results, fall back to original results
                    if not results and original_results:
                        logger.debug("Filtered too aggressively, falling back to original results")
                        results = original_results
                    
                    result = results[0]
                    location_details = {
                        "city": result.get("name", city),
                        "state": result.get("admin1"),
                        "country": result.get("country"),
                        "country_code": result.get("country_code")
                    }
                    
                    logger.debug(f"Selected location: {location_details['city']}, " +
                                f"{location_details['state'] or 'N/A'}, " +
                                f"{location_details['country'] or 'N/A'}")
                    
                    return result["latitude"], result["longitude"], location_details
                    
        except aiohttp.ClientError:
            raise ValueError("Error connecting to location service. Please try again.")
        except asyncio.TimeoutError:
            raise ValueError("Location service timed out. Please try again.")
    
    def _get_weather_code_description(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: "clear skys",
            1: "mainly clear skys",
            2: "partly cloudy skys",
            3: "overcast skys",
            45: "foggy conditions",
            48: "foggy conditions with frost",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "heavy drizzle",
            56: "light freezing drizzle",
            57: "heavy freezing drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "light snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "light rain showers",
            81: "moderate rain showers",
            82: "heavy rain showers",
            85: "light snow showers",
            86: "heavy snow showers",
            95: "thunderstorms",
            96: "thunderstorms with light hail",
            99: "thunderstorms with heavy hail"
        }
        return codes.get(code, "unknown conditions")
    
    async def execute(self, args: List[str]) -> str:
        """Execute weather tool with ordered arguments."""
        try:
            # Convert args list to dict based on defined order
            arg_dict = {}
            for i, value in enumerate(args):
                if value and i < len(self.args):  # Only include non-empty args
                    arg_dict[self.args[i]] = value

            # Get coordinates and location details
            lat, lon, location_details = await self._get_coordinates(
                arg_dict["city"],
                arg_dict.get("state"),
                arg_dict.get("country")
            )
            
            # Get weather data
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current": [
                        "temperature_2m",
                        "weather_code",
                        "wind_speed_10m"
                    ],
                    "daily": [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max"
                    ],
                    "temperature_unit": self.config["settings"]["units"]["temperature"],
                    "wind_speed_unit": self.config["settings"]["units"]["wind_speed"],
                    "precipitation_unit": self.config["settings"]["units"]["precipitation"],
                    "timezone": "auto",
                    "forecast_days": 1
                }
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise ValueError("Unable to get weather data. Please try again later.")
                    
                    data = await response.json()
                    
                    current = data["current"]
                    daily = data["daily"]
                    
                    conditions = self._get_weather_code_description(current['weather_code'])
                    units = self.config["settings"]["units"]
                    wind_speed_unit = "miles per hour" if units["wind_speed"] == "mph" else "kilometers per hour"
                    
                    # Format location string based on available details
                    location_str = location_details['city']
                    if location_details.get('state'):
                        location_str += f", {location_details['state']}"
                    if location_details.get('country'):
                        location_str += f", {location_details['country']}"
                    
                    return (
                        f"In {location_str}, the current temperature is {round(current['temperature_2m'])} degrees {units['temperature']} "
                        f"with {conditions}. The wind speed is {round(current['wind_speed_10m'])} {wind_speed_unit}. "
                        f"Today's high will be {round(daily['temperature_2m_max'][0])} degrees {units['temperature']} "
                        f"and the low will be {round(daily['temperature_2m_min'][0])} degrees {units['temperature']}. "
                        f"There is a {round(daily['precipitation_probability_max'][0])} percent chance of precipitation."
                    )
                    
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Error getting weather data: {str(e)}")
            return "An error occurred while getting the weather data. Please try again later."
