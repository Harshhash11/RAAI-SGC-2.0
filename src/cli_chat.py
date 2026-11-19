import argparse
from pathlib import Path
from src.hybrid import load_data,HybridRetriever
from src.answers import extractive_answer
from src.llm import generate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--question',required=True)
    p.add_argument('--mode',choices=['vector','unweighted','hybrid'],default='hybrid')
    p.add_argument('--backend',choices=['tfidf','semantic'],default='tfidf')
    p.add_argument('--llm-model',default='')
    args=p.parse_args()
    sources,edges,_=load_data(Path(__file__).resolve().parents[1]/'data/processed/master.csv')
    ret=HybridRetriever(sources,edges,args.backend)
    results,debug=ret.search(args.question,top_k=3,mode=args.mode)
    output=generate(args.question,results,args.llm_model) if args.llm_model else extractive_answer(args.question,results)
    print(f"MODE: {output['mode']} | GRAPH ACTIVATED: {debug['graph_activated']}")
    print(output['answer'])
    for c in output['citations']:print(f"SOURCE: {c['title']} — {c['url']}")
if __name__=='__main__':main()
