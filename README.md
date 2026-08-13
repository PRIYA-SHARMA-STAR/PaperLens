# PaperLens
PaperLens — AI-powered research paper discovery, summarization, and insights

Solution
The system combines several powerful NLP tools to build an intelligent research paper search engine over 15,000 ML ArXiv papers.

Semantic Embedding — Instead of matching keywords, the system converts both papers and search queries into numerical representations that capture their meaning, so conceptually similar papers surface even when they use different words.

FAISS Indexing — All paper embeddings are indexed and normalized, enabling fast and accurate similarity search across the entire dataset in milliseconds.

Summarization — Each retrieved paper's abstract is automatically condensed into a short, readable summary so you can quickly grasp the core idea without reading the full abstract.

Keyword Extraction — The most important phrases from each paper are pulled out automatically, giving you an at-a-glance view of what the paper is about.

Named Entity Recognition — A domain-specific scientific NER model identifies and extracts technical terms from each retrieved abstract, highlighting the methods, concepts, and techniques the paper deals with. A custom gazetteer further extends this by catching domain-specific terms the model might miss, ensuring important ML terminology is never overlooked.
