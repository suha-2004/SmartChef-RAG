import requests
import json
import time

BASE_URL = "https://www.themealdb.com/api/json/v1/1"

all_meals = []
meal_ids = set()

print("Fetching recipes from TheMealDB...")
print()

# Get meals alphabetically
for letter in "abcdefghijklmnopqrstuvwxyz":

    print(f"Fetching recipes starting with '{letter}'...")

    url = f"{BASE_URL}/search.php?f={letter}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        meals = data.get("meals")

        if meals:
            for meal in meals:

                meal_id = meal.get("idMeal")

                # Avoid duplicate recipes
                if meal_id and meal_id not in meal_ids:
                    meal_ids.add(meal_id)

                    all_meals.append(meal)

    except Exception as e:
        print(f"Error for letter {letter}: {e}")

    time.sleep(0.2)


print()
print(f"Total recipes found: {len(all_meals)}")

# Save recipes
with open("recipes_api.json", "w", encoding="utf-8") as file:
    json.dump(all_meals, file, indent=4, ensure_ascii=False)

print()
print("Recipes saved successfully!")
print("File created: recipes_api.json")