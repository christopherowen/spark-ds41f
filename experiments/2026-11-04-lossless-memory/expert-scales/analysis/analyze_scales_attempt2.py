import hashlib,importlib.util,json,os,pathlib,subprocess,sys
home=pathlib.Path.home();base=home/"tl-logs/lossless-memory/recovery";root=home/"projects/spark-ds41f-lossless-memory-recovery"
snapshot=base/"scales-round4-snapshot/dgx1/projects/spark-ds41f";output=base/"scale-analysis-attempt2";output.mkdir(exist_ok=False)
os.chdir(root);sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location("component_runner",root/"experiments/2026-11-04-lossless-memory/run_component.py");runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
receipt=json.loads((snapshot/"results/private/lossless-memory-scales-round4/receipt.json").read_text())
arm=root/"experiments/2026-11-04-lossless-memory/expert-scales";release=json.loads((arm/"release-tp3.json").read_text());job=release["jobs"][0]
audit=[]
for number,name in enumerate(job["suite"]):
 suffix="" if number==0 else "-"+name
 for candidate in job["arms"]:
  directory=snapshot/"results/private/suites"/(release["run"]+suffix)/candidate["label"]
  starts=runner.native_suite_gate(directory,runner.suite.load(name),candidate["config"],receipt["commit"],receipt["images"])
  audit.append({"label":candidate["label"],"suite":name,"native_suite_and_conditions_valid":True,"starts":starts})
for name in ["kernel","full-gate"]:runner.native_kernel_gate(receipt[name]["native"],receipt["images"],inputs=receipt[name]["inputs_sha256"])
for label,row in receipt["model_identity"].items():
 path=snapshot/row["path"];data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==row["sha256"] and row["passed"]
 reference=receipt["model_reference"];candidate=snapshot/"results/private/suites"/release["run"]/label/"model.json.gz"
 runner.native_model_gate(json.loads(data),candidate,reference)
(output/"native-audit.json").write_text(json.dumps({"execution_commit":receipt["commit"],"suites":audit,"all_native_image_model_gates_passed":True},indent=2)+"\n")
command=[sys.executable,str(base/"analyze_component_performance.py"),"--snapshot",str(snapshot),"--component","expert-scales","--receipt","results/private/lossless-memory-scales-round4/receipt.json","--reference-repeat-root",str(root/"experiments/2026-11-04-lossless-memory/runs/recorded-serving-references/results/private/suites"),"--out",str(output/"performance.json")]
print(json.dumps({"analysis_command":command}),flush=True)
subprocess.run(command,cwd=root,check=True)
