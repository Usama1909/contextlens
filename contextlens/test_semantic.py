from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

m = SentenceTransformer('all-MiniLM-L6-v2')

msgs = [
    'Regime:CRISIS Score:-45.4 Top:GLD',
    'Regime:CRISIS Score:-45.5 Top:GLD',
    'Regime:CRISIS Score:-48.8 Top:GLD',
    'Regime:NORMAL Score:-11.9 Top:NVDA',
    'The user wants to reset their password',
]

embeddings = m.encode(msgs)
base = embeddings[0]

print('Similarity to first message:')
for i, msg in enumerate(msgs):
    sim = cosine_similarity([base], [embeddings[i]])[0][0]
    print(f'  {sim:.3f} | {msg}')