#!/usr/bin/env python3
"""Cross-check every cited core arXiv title using the live DataCite registration API."""
from pathlib import Path
from datetime import date
import difflib,json,re,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def normal(s):return re.sub(r'[^a-z0-9]','',s.lower())
def main():
    records=json.loads((ROOT/'sources/evidence.json').read_text(encoding='utf8'))
    records+=list(json.loads((ROOT/'sources/core-reading.json').read_text(encoding='utf8')).values())
    expected={re.sub(r'v\d+$','',r['arxiv']):r['title'] for r in records}
    out=[]
    for aid,title in expected.items():
        endpoint='https://api.datacite.org/dois/10.48550/arXiv.'+aid
        response=None
        for attempt in range(3):
            result=subprocess.run(['curl','-fsS','--max-time','30',endpoint],capture_output=True,text=True)
            if result.returncode==0:
                response=json.loads(result.stdout);break
            time.sleep(3*(attempt+1))
        if response is None:raise RuntimeError(f'No verified metadata for {aid}; stopped, not silently skipped')
        a=response['data']['attributes'];actual=a['titles'][0]['title']
        score=difflib.SequenceMatcher(None,normal(title),normal(actual)).ratio()
        if score<.92:raise RuntimeError(f'Title mismatch {aid}: {title!r} / {actual!r}')
        out.append({'arxiv':aid,'expected_title':title,'api_title':actual,'doi':response['data']['id'],'publication_year':a['publicationYear'],'endpoint':endpoint,'checked_on':date.today().isoformat(),'title_similarity':score,'creators':a.get('creators',[])})
        print('verified',aid,actual,flush=True);time.sleep(.6)
    target=ROOT/'sources/primary/datacite-verified.json'
    temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(target)
    print('Verified all',len(out),'unique core sources.')
if __name__=='__main__':main()
