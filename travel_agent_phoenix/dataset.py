import pandas as pd
from phoenix.client import Client

client = Client()

data = [
    {
        "input": "Plan a 7-day trip to Italy focused on art, history, and local food. Include essential travel info, a budget estimate, and key attractions in Rome, Florence, and Venice.",
        "reference_output": """
Here’s a detailed 7-day itinerary for Italy focused on art, history, and local cuisine, covering Rome, Florence, and Venice. Estimated costs are included for budgeting.

Day 1–3: Rome
- Visit the Colosseum, Roman Forum, and Palatine Hill (Cost: $30)
- Vatican Museums and Sistine Chapel (Cost: $20)
- Explore Pantheon, Trevi Fountain, and Piazza Navona
- Food experiences in Trastevere including pasta and gelato (Meals: ~$40/day)

Accommodation: $180/night
Transportation: Metro and walking ($10/day)

Day 4–5: Florence
- Uffizi Gallery and Accademia (Cost: $35)
- Florence Cathedral and Duomo climb (Cost: $20)
- Tuscan wine tasting tour (Cost: $60)
- Local trattorias and street food ($35/day)

Accommodation: $160/night

Day 6–7: Venice
- St. Mark’s Basilica and Doge’s Palace (Cost: $30)
- Gondola ride (shared) (Cost: $40)
- Island visit to Murano and Burano
- Cicchetti food crawl ($30/day)

Accommodation: $190/night

Estimated Total Budget (7 days): $2,400–$2,700
Travel Tips: Use trains between cities and book museums in advance.
"""
    },

    {
        "input": "Create a 4-day itinerary for Seoul centered on K-pop, fashion districts, and street food. Include transportation tips and a mid-range budget.",
        "reference_output": """
Here’s a 4-day Seoul itinerary focused on K-pop culture, fashion hubs, and street food with mid-range budgeting.

Day 1:
- Visit Gangnam and K-Star Road
- Lunch at local Korean BBQ ($25)
- Evening at COEX Mall and SMTOWN Museum ($20)

Day 2:
- Hongdae fashion streets and street food crawl ($20)
- K-pop dance class experience ($40)
- Live music cafes in Hongdae

Day 3:
- Myeongdong shopping district
- Street food dinner ($15)
- N Seoul Tower night view ($15)

Day 4:
- Traditional contrast at Bukchon Hanok Village
- Gwangjang Market food tour ($20)

Accommodation: $120/night
Transport: T-money card (~$5/day)

Estimated Total Budget: $850–$1,000
"""
    }
]

dataset_df = pd.DataFrame(data)

dataset = client.datasets.create_dataset(
    dataframe=dataset_df,
    name="travel-questions-with-references_v8",
    input_keys=["input"],
    output_keys=["reference_output"],
)