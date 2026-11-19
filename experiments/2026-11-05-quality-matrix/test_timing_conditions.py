"""Reject missing fan control before launching a model benchmark."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import measure
from timing_conditions import fan_curve_ready

class TimingConditions(unittest.TestCase):
    def test_active_and_successfully_started_services_qualify(self):
        self.assertTrue(fan_curve_ready([{'node':'a','service':'active'},{'node':'b','service':'inactive','started':True}],['a','b']))
    def test_absent_failed_duplicate_incomplete_and_disabled_records_do_not(self):
        for records in [[],[{'node':'a','service':'absent'}],[{'node':'a','service':'inactive','started':False}],[{'node':'b','service':'active'}],[{'node':'a','service':'active'}]*2]:
            with self.subTest(records=records):self.assertFalse(fan_curve_ready(records,['a']))
    def test_missing_service_stops_before_any_model_request_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'answers.json';watch=MagicMock();watch.thermal_summary.return_value={};watch.close.return_value={}
            argv=['measure.py','--profile','candidate.json','--benchmark','ruler','--cache',d,'--private',d,'--out',str(out)]
            with patch('sys.argv',argv),patch.object(measure,'read_cluster',return_value={'nodes_config':'nodes','host':{'memguard_gib':1}}),patch.object(measure,'read_json',return_value={'nodes':[{'name':'a'}]}),patch.object(measure.bench,'MemoryWatch',return_value=watch),patch.object(measure.bench,'fan_control_enabled',return_value=True),patch.object(measure.bench,'cool_nodes',return_value=[]),patch.object(measure.bench,'start_fan_curve',return_value=[{'node':'a','service':'absent','started':False}]),patch.object(measure.bench,'stop_fan_curve',return_value=[]),patch.object(measure.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'not installed'):measure.main()
                launch.assert_not_called();watch.close.assert_called_once()
            self.assertFalse(out.exists());self.assertTrue(json.loads(out.with_name('answers-conditions.json').read_text())['fan_warnings'])

    def test_longbench_adapter_marks_stock_fan_history_exploratory(self):
        import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'source.json').write_text(json.dumps({'benchmark':'longbench'}))
            c={'halted':[],'fan_stop_failures':[],'thermal':{'a':{'thermal_slowdown_ms':0}},'fan_curve':[{'node':'a','service':'absent'}]}
            for label in summarize.LABELS:(root/('answers-'+label+'-conditions.json')).write_text(json.dumps(c))
            def delegate(*args,**kwargs):(root/'comparison.json').write_text(json.dumps({'complete':True,'promotion_qualified':False}))
            with patch('sys.argv',['summarize.py',str(root)]),patch.object(summarize.subprocess,'run',side_effect=delegate):summarize.main()
            self.assertFalse(json.loads((root/'comparison.json').read_text())['timing_qualified'])

    def test_incomplete_longbench_has_no_timing_qualification(self):
        import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'source.json').write_text(json.dumps({'benchmark':'longbench'}))
            def delegate(*args,**kwargs):(root/'comparison.json').write_text(json.dumps({'complete':False}))
            with patch('sys.argv',['summarize.py',str(root)]),patch.object(summarize.subprocess,'run',side_effect=delegate):summarize.main()
            self.assertFalse(json.loads((root/'comparison.json').read_text())['timing_qualified'])

if __name__=='__main__':unittest.main()
