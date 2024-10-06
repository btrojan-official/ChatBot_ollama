import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from uuid import uuid4
from langchain_community.vectorstores import FAISS

urls = [
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://eucrim.eu/news/new-rules-for-crypto-assets-in-the-eu/",
    "https://www.crypto-law.us/",
    "https://freemanlaw.com/legal-issues-surrounding-cryptocurrency/",
    "https://www.globallegalinsights.com/practice-areas/blockchain-laws-and-regulations/usa/",
    "https://stevenscenter.wharton.upenn.edu/publications-50-state-review/",
    "https://www.reuters.com/world/us/us-securities-regulator-urges-against-crypto-bill-adoption-2024-05-22/",
    "https://www.coursera.org/articles/how-does-cryptocurrency-work",
    "https://www.nerdwallet.com/article/investing/cryptocurrency",
    "https://www.investopedia.com/terms/b/blockchain.asp",
    "https://www.ibm.com/topics/blockchain",
    "https://www.tripwire.com/state-of-security/key-takeaways-crypto-crime-update",
    "https://go.chainalysis.com/crypto-crime-2024.html",
    "https://fortune.com/crypto/2024/08/23/when-it-comes-to-crime-crypto-is-a-powerful-tool-for-law-enforcement/",
    "https://coinranking.com/coins/shitcoins",
    "https://coinmarketcap.com/watchlist/60812f64f27fc650843d8749/",
    "https://aibc.world/learn-crypto-hub/ultimate-shitcoin-guide/",
    "https://www.scotthugheslaw.com/documents/CRYPTOCURRENCY-REGULATIONS-AND-ENFORCEMENT-IN-THE-US-2.pdf",
    "https://www.nortonrosefulbright.com/en/knowledge/publications/184ac2f1/2024-fintech-outlook",
    "https://www.ncsl.org/financial-services/cryptocurrency-digital-or-virtual-currency-and-digital-assets-2024-legislation",
    "https://www.coindesk.com/policy/2024/08/31/a-us-crypto-bills-2024-chances/",
    "https://www.coinbase.com/pl/learn/crypto-basics/understanding-crypto-taxes",
    "https://koinly.io/cryptocurrency-taxes/",
    "https://www.investopedia.com/tech/taxes-and-crypto/",
    "https://www.fool.com/investing/stock-market/market-sectors/financials/cryptocurrency-stocks/crypto-taxes/"
]

# Load documents
docs = []
for url in urls:
    try:
        docs.append(WebBaseLoader(url).load())
    except:
        print(f"Skipping {url}")
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400, chunk_overlap=80
)

doc_splits = text_splitter.split_documents(docs_list)
doc_splits_ids = [str(uuid4()) for _ in range(len(doc_splits))]

print("Loading embeddings model")

from langchain_nomic.embeddings import NomicEmbeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

print("Populating database")

# Initialize FAISS vectorstore with document splits and embeddings
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings,
)

vectorstore.save_local("my_faiss_store")
