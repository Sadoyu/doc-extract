import sys, json
sys.path.insert(0, '/Users/saikiranchandolu/Workspace/Github/doc-extract')
from src import main
from src.LenientOllamaFormatHandler import parse_output

metrics_config = main.load_metrics_config('metrics.yaml')
text = 'NAV per share: $12.34. AUM: $1,000,000. Quarterly return: 2.5%.'

print('==== rule_based_extract ====')
rb = main.rule_based_extract(text, metrics_config)
print(json.dumps(rb, indent=2))

print('\n==== parse_output malformed JSON ====')
po = parse_output('{"fund_name": "Global Fund",')
print(repr(po))

print('\n==== getExtractedMetrics with fake dict extraction ====')
class FakeResult:
    def __init__(self, extractions):
        self.extractions = extractions

fake_extractions = [
    {
        'extraction_class': 'fund_aum',
        'extraction_text': '$1,200,000,000',
        'attributes': {'metric_name': 'fund_aum', 'value': '$1,200,000,000'}
    },
    {
        'extraction_class': 'nav_per_share',
        'extraction_text': "$12.34",
        'attributes': {'metric_name': 'nav_per_share', 'value': '$12.34'}
    },
    # stringified dict form
    {
        'extraction_class': '',
        'extraction_text': "{'extraction_class': 'quarterly_return', 'extraction_text': '3.5%', 'attributes': {'value': '3.5%'}}",
        'attributes': {}
    }
]
fake_result = FakeResult(fake_extractions)
print(json.dumps(main.getExtractedMetrics(fake_result), indent=2))

print('\n==== getExtractedMetrics with fake object-like extraction ====')
class FakeExtractionObj:
    def __init__(self, cls, text, attrs=None):
        self.extraction_class = cls
        self.extraction_text = text
        self.attributes = attrs or {}
        self.description = ''

fake_obj_result = FakeResult([
    FakeExtractionObj('fund_aum', '$2,350,000,000', {'metric_name':'fund_aum','value':'$2,350,000,000'}),
    FakeExtractionObj('nav_per_share', '$45.67', {'metric_name':'nav_per_share','value':'$45.67'}),
])
print(json.dumps(main.getExtractedMetrics(fake_obj_result), indent=2))

