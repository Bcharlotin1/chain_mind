import subprocess
import sys

print("Step 1/2: Scraping LangChain docs...")
subprocess.run([sys.executable, "src/scraper.py"], check=True)

print("\nStep 2/2: Building vector index...")
subprocess.run([sys.executable, "src/vector_store.py"], check=True)

print("\nSetup complete! Run the app with: streamlit run app.py")
