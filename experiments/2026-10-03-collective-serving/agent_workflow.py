"""Synthetic read-only tool workflow; no real orders, bookings or messages."""
import argparse, hashlib, json, pathlib, re, time, urllib.request, uuid
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
base=a.url.rstrip('/')
prompt='Look up order A-17, obtain shipping quotes using its actual destination and weight, then choose the cheapest service that delivers within two days. State the selected service and its total price in EUR. Do not book anything.'
tools=[{'type':'function','function':{'name':'lookup_order','description':'Read the order details before requesting shipping quotes.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id'],'additionalProperties':False}}},{'type':'function','function':{'name':'quote_shipping','description':'Read available shipping services for an order destination and weight.','parameters':{'type':'object','properties':{'destination':{'type':'string'},'weight_kg':{'type':'number'}},'required':['destination','weight_kg'],'additionalProperties':False}}}]
def post(body):
 request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
report={'url':base,'synthetic':True,'workload_sha256':hashlib.sha256(json.dumps([prompt,tools],sort_keys=True).encode()).hexdigest(),'runs':[]}
try:
 for trial in range(a.repeats):
  messages=[{'role':'user','content':prompt}];calls=[];turns=[];salt=uuid.uuid4().hex;started=time.monotonic();lookup_turn=None;final=''
  for turn in range(5):
   reply=post({'model':'deepseek-v4.1-flash','messages':messages,'tools':tools,'tool_choice':'auto','temperature':0,'seed':42,'max_tokens':256,'chat_template_kwargs':{'thinking':False},'cache_salt':salt})
   message=reply['choices'][0]['message'];turns.append({'message':message,'usage':reply.get('usage')});messages.append(message)
   if not message.get('tool_calls'):
    final=message.get('content') or '';break
   for call in message['tool_calls']:
    name=call['function']['name'];args=json.loads(call['function']['arguments']);calls.append(name)
    if name=='lookup_order':
     assert args=={'order_id':'A-17'},args
     assert calls==['lookup_order'],calls
     lookup_turn=turn;result={'order_id':'A-17','destination':'Prague','weight_kg':2.5}
    elif name=='quote_shipping':
     assert lookup_turn is not None and turn>lookup_turn,'quote requested before reading order result'
     assert args=={'destination':'Prague','weight_kg':2.5},args
     result={'currency':'EUR','offers':[{'service':'Economy','total':4,'delivery_days':5},{'service':'Express','total':9,'delivery_days':2},{'service':'Priority','total':15,'delivery_days':1}]}
    else:raise AssertionError(f'unexpected tool {name}')
    messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
  ok=calls==['lookup_order','quote_shipping'] and 'express' in final.lower() and bool(re.search(r'\b9(?:[.,]0+)?\b',final))
  report['runs'].append({'trial':trial,'ok':ok,'seconds':time.monotonic()-started,'calls':calls,'turns':turns,'final':final})
  print(json.dumps({'trial':trial,'ok':ok,'seconds':round(report['runs'][-1]['seconds'],3),'calls':calls}),flush=True)
  assert ok,final
except BaseException as e:
 report['error']=repr(e);raise
finally:
 pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
