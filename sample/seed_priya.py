# seed_priya.py
"""
Pre-load some memories for 'priya' — a lifestyle-oriented user who loves
shopping, reading, and exotic travel. She has a wealthy husband and two kids.
Run once:  python seed_priya.py
"""

from client import SmritikoshClient

client = SmritikoshClient(username="admin", password="changeme123")

USER = "priya"

memories = [
    "My name is Priya and I'm a homemaker who loves discovering new things every day.",
    "I am passionate about fashion and shopping. I follow several luxury brands like Chanel, Bottega Veneta, and Loro Piana.",
    "I read at least two books a month. I enjoy literary fiction and travel memoirs — authors like Chimamanda Ngozi Adichie and Pico Iyer are favourites.",
    "My husband Rohan is a successful investment banker. He is very supportive and enjoys spoiling me with travel and experiences.",
    "We have two kids — Aanya who is 8 and Kabir who is 5. Aanya is into art and Kabir is obsessed with dinosaurs.",
    "I love planning family vacations to exotic destinations. We recently came back from the Maldives and are planning a trip to Patagonia next winter.",
    "I keep a running wishlist of destinations: Amalfi Coast, Bhutan, Kyoto during cherry blossom season, and the Faroe Islands.",
    "I have a personal stylist and usually refresh my wardrobe each season. I prefer investment pieces over fast fashion.",
    "Rohan does not follow fashion at all — he relies on me to pick his outfits for dinners and events.",
    "I recently joined a book club with four other women from our neighbourhood. We meet every third Sunday over brunch.",
    "My guilty pleasure is spending Sunday mornings browsing Net-a-Porter and Mytheresa while the kids watch cartoons.",
    "We stayed at Soneva Fushi in the Maldives last month — the underwater restaurant was the highlight for Aanya.",
    "I am trying to get Kabir into reading but he only wants stories about T-Rex. Any suggestion for books that mix adventure with dinosaurs would be great.",
    "Rohan handles all our investments and finances. I manage the household budget, which is fairly generous, but I still love a good sale.",
    "I want to take the kids to Japan next year. I am researching kid-friendly ryokans and am excited about the food and temples.",
]

print(f"Seeding {len(memories)} memories for user '{USER}'...\n")

for i, text in enumerate(memories, 1):
    result = client.remember(USER, text)
    importance = result.get("importance_score", 0)
    facts = result.get("facts_extracted", 0)
    print(f"  [{i:2d}] importance={importance:.2f}  facts={facts}  \"{text[:60]}...\"")

print("\nDone. Run 'python chatbot.py' to chat with Priya's memory.")
