import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eval_data_builder',ROOT/'tools/build_eval_data.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class EvalDataBuilderTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(sample_id=f'g{i}',query=f'old{i}',prompt=[],target_bbox=[1,2,3,4]) for i in range(18)]
        self.patches=[dict(groundbench_sample_id=f'g{i}',degraded_sample_id=f'd{i}',
            old_query=f'old{i}',new_query=f'new{i}') for i in range(17)]

    def test_patch_exactly_seventeen_queries(self):
        result=b.replace_queries(self.rows,self.patches,'groundbench')
        self.assertEqual(sum(a!=c for a,c in zip(self.rows,result)),17)
        self.assertEqual(result[-1],self.rows[-1])
        self.assertTrue(all(a['target_bbox']==c['target_bbox'] for a,c in zip(self.rows,result)))
        self.assertEqual(self.rows[0]['query'],'old0')

    def test_reverse_roundtrip(self):
        updated=b.replace_queries(self.rows,self.patches,'groundbench')
        self.assertEqual(b.replace_queries(updated,self.patches,'groundbench',True),self.rows)

    def test_already_corrected_is_unchanged(self):
        updated=b.replace_queries(self.rows,self.patches,'groundbench')
        self.assertEqual(b.replace_queries(updated,self.patches,'groundbench'),updated)

    def test_unknown_query_rejected(self):
        self.rows[0]['query']='unrelated annotation'
        with self.assertRaises(ValueError):b.replace_queries(self.rows,self.patches,'groundbench')

    def test_missing_patch_target_rejected(self):
        with self.assertRaises(ValueError):b.replace_queries(self.rows[1:],self.patches,'groundbench')

    def test_nonempty_prompt_rejected(self):
        self.rows[0]['prompt']=['unverified prompt']
        with self.assertRaises(ValueError):b.replace_queries(self.rows,self.patches,'groundbench')

    def test_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json'
            path.write_text(json.dumps({'samples':[]}),encoding='utf-8')
            self.assertEqual(b.read(path),{'samples':[]})
            path.write_text('{"n":1}\n{"n":2}\n',encoding='utf-8')
            self.assertEqual(b.read(path),[{'n':1},{'n':2}])

    def test_evaluation_annotation_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';data.mkdir()
            (data/'degraded.json').write_text(json.dumps(self.rows),encoding='utf-8')
            self.assertEqual(b.read_degraded_annotations(tmp),self.rows)

    def test_legacy_annotation_filename_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';data.mkdir()
            (data/'degraded_v2.json').write_text(json.dumps(self.rows),encoding='utf-8')
            self.assertEqual(b.read_degraded_annotations(tmp),self.rows)

    def test_evaluation_annotation_filename_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';data.mkdir()
            (data/'degraded.json').write_text(json.dumps(self.rows),encoding='utf-8')
            (data/'degraded_v2.json').write_text('[]',encoding='utf-8')
            self.assertEqual(b.read_degraded_annotations(tmp),self.rows)

    def test_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ['../escape.jpg','/absolute.jpg','C:/absolute.jpg']:
                with self.assertRaises(ValueError):b.under(tmp,path)

    def test_pair_source_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for modality in ['visible','infrared']:
                path=root/'HDRT'/modality/'x.jpg';path.parent.mkdir(parents=True);path.touch()
            self.assertEqual(len(b.dv_pair(root,'x.jpg')),2)
            for modality in ['visible','infrared']:
                path=root/'LLVIP'/modality/'x.jpg';path.parent.mkdir(parents=True);path.touch()
            with self.assertRaises(ValueError):b.dv_pair(root,'x.jpg')
            self.assertTrue(all('HDRT' in p for p in b.dv_pair(root,'x.jpg','HDRT')))


if __name__=='__main__':unittest.main()
