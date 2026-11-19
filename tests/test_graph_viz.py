import unittest
from pathlib import Path
from src.hybrid import load_data, HybridRetriever
from src.graph_viz import build_subgraph, make_figure, source_links, load_graph_attributes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/master.csv'

class GraphViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.edges, _ = load_data(DATA)
        cls.searcher = HybridRetriever(cls.sources, cls.edges, backend='tfidf')
        cls.hits, cls.diag = cls.searcher.search('What is Kubernetes?',top_k=3,mode='hybrid')

    def test_actual_graph_edges(self):
        graph = build_subgraph(self.sources, self.edges, self.hits, max_edges=40, csv_path=DATA)
        self.assertTrue(graph['edges'])
        self.assertLessEqual(len(graph['edges']),40)
        self.assertEqual(graph['source_ids'][0],self.hits[0]['source_id'])
        for e in graph['edges']:
            self.assertTrue(any(e['term']==t and abs(e['weight']-w)<1e-12 for t,w in self.edges[e['source_id']]))
            self.assertTrue(e['from'].startswith('source:'))
            self.assertTrue(e['to'].startswith('term:'))

    def test_graph_weight_arithmetic(self):
        g=build_subgraph(self.sources,self.edges,self.hits,max_edges=80,csv_path=DATA)
        self.assertTrue(any(e['markov'] is not None for e in g['edges']))
        for e in g['edges']:
            if e['markov'] is not None:
                self.assertAlmostEqual(e['weight'],e['markov']*(1+e['tfidf']),places=8)

    def test_plotly_and_links(self):
        g=build_subgraph(self.sources,self.edges,self.hits,max_edges=24,csv_path=DATA)
        fig=make_figure(g)
        self.assertGreater(len(fig.data),0)
        self.assertIn('Kubernetes',source_links(g,self.sources))
        self.assertIn('https://',source_links(g,self.sources))

    def test_empty_and_limits(self):
        self.assertFalse(build_subgraph(self.sources,self.edges,[])['nodes'])
        g=build_subgraph(self.sources,self.edges,self.hits,max_edges=999)
        self.assertLessEqual(len(g['edges']),120)
        self.assertTrue(make_figure({'nodes':[],'edges':[],'source_ids':[]}).layout)

if __name__=='__main__': unittest.main()
