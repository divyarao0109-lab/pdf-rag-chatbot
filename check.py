import importlib
for m in ["numpy", "pyarrow", "streamlit", "pypdf", "torch", "sentence_transformers",
          "langchain", "langchain_community", "langchain_huggingface",
          "langchain_ollama", "ollama", "rank_bm25"]:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        print("FAIL", m, "-", str(e)[:90])