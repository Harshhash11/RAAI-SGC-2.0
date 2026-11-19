import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid import HybridRetriever, load_data, canonical


def source(sid,title,split='train',text=''):
    return {'source_id':sid,'source_title':title,'source_url':'https://example.org/'+sid,
            'domain':'cloud_devops','split':split,'source_text':text,
            'review_status':'UNREVIEWED','license_status':'REVIEW_CC_BY_SA'}


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources={'a':source('a','Kubernetes',text='Kubernetes container orchestration'),
                     'b':source('b','Container',text='A container isolates an app'),
                     'c':source('c','Authorization','test',text='authorization rules')}
        cls.edges={'a':[('container',0.2)],'b':[],'c':[]}
        cls.engine=HybridRetriever(cls.sources,cls.edges)
    def test_canonical(self):
        self.assertEqual(canonical('Container (computing)'),'container')
    def test_title(self):
        r,_=self.engine.search('What is Kubernetes?',mode='vector')
        self.assertEqual(r[0]['source_id'],'a')
    def test_graph_activates(self):
        r,d=self.engine.search('What is Kubernetes?',mode='hybrid',top_k=3)
        self.assertTrue(d['graph_activated'])
        self.assertGreater(next(x for x in r if x['source_id']=='b')['graph_score'],0)
    def test_unweighted(self):
        _,d=self.engine.search('What is Kubernetes?',mode='unweighted')
        self.assertTrue(d['graph_activated'])
    def test_no_cross_split_graph(self):
        r,d=self.engine.search('What is Kubernetes?',mode='hybrid',split='test')
        self.assertFalse(d['graph_activated'])
        self.assertTrue(all(x['split']=='test' for x in r))
    def test_unknown(self):
        with self.assertRaises(ValueError): self.engine.search('abc',mode='invalid')
    def test_no_results_empty(self):
        r,d=self.engine.search('  ')
        self.assertEqual(r,[])
    def test_dataset(self):
        p=Path(__file__).resolve().parents[1]/'data'/'processed'/'master.csv'
        s,e,q=load_data(p)
        self.assertEqual((len(s),q),(145,290))
        self.assertEqual(sum(map(len,e.values())),4771)

if __name__=='__main__':unittest.main()
