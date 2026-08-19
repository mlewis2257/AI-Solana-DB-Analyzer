import wikipediaapi
import os

wiki = wikipediaapi.Wikipedia(
    user_agent="AIEngineering-RAG-Project/1.0", language="en")

topics = ["Neurons", "Synapses", "Cerebral Cortex",
          "Hippocampus", "Neuroplasticity", "Action Potential", "Imagination", ]

os.makedirs("data", exist_ok=True)

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        filepath = f"data/{topic.replace(' ','_')}.txt"
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(page.text)
        print(f"SAVED: {filepath} ({len(page.text)} chars)")
    else:
        print(f"NOT FOUND: {topic}")
