//! Ignored, explicitly admitted real-component witness, not model inference.
//! Storage keeps actual typed leaves. GPU arithmetic uses existing components;
//! the two small outer BF16 operations preserve pinned HF cast boundaries.
use super::*;
use crate::{format::attrs, model_collection};
use crate::models::nemotron_h::{attention, ingress::{native, packing}, mamba2, pile as native_pile};
use anybytes::Bytes;
use anyhow::{Context, Result as AnyResult, ensure};
use ed25519_dalek::SigningKey;
use safetensors::{Dtype as SourceDtype, SafeTensors};
use serde_json::{Value, json};
use std::{fs, io::{Read, Write}, path::{Path, PathBuf}, process::{Command, Stdio}};
use triblespace::{prelude::*, core::{blob::encodings::tensor::{TensorView, elements::BF16},
    collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt}}};

const CAP: usize = 32 << 20; // Native + independent reference each32MiB.
const WORK: usize = 64 << 20;
const SHARDS: [(usize, &str, u64, &str); 2] = [
    (2,"model-00003-of-00088.safetensors",109705480,"18829df5916e1e388046848b4af922e7a15990957b93269a58924df4806069eb"),
    (7,"model-00008-of-00088.safetensors",35661352,"5ae1f86003d237659f6fb990ce55e2f4adeb9c0e81ac484b63de066b60746418"),
];
const PLANS: [(&str, &[usize]); 5] = [
    ("T1", &[1]), ("T3", &[3]), ("split", &[2,1]), ("steps", &[1,1,1]), ("reset", &[3]),
];

#[test]
fn fixture_contract_is_bounded_and_not_full_model() {
    assert_eq!(SHARDS.iter().map(|s|s.2).sum::<u64>(),145_366_832);
    assert_eq!(SHARDS.map(|s|s.0),[2,7]);
    assert_eq!(PLANS.iter().map(|(_,p)|p.iter().sum::<usize>()).collect::<Vec<_>>(),[1,3,3,3,3]);
    // Final F32 recurrent state/BF16 tail in five arms plus generous stage,
    // projection diagnostic and metadata allowance, per implementation.
    let bound=5*(4_194_304+61_440)+13*(18_560+10_240+2*8192+2*4096)*2+4*3*4096*2+(2<<20);
    assert!(bound<CAP);
    assert!(2*CAP<=64<<20);
}

fn sha256(bytes: &[u8]) -> String {
    let mut child=Command::new("sha256sum").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let out=child.wait_with_output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap().split_whitespace().next().unwrap().to_owned()
}
fn json_write(path: &Path, value: &Value) { fs::write(path,serde_json::to_vec_pretty(value).unwrap()).unwrap(); }
fn variable(name: &str) -> PathBuf { std::env::var_os(name).expect(name).into() }

#[test]
#[ignore = "CPU: explicit two pinned shards and NEW scratch destination; no live pile/key"]
fn import_actual_components() -> AnyResult<()> {
    let source=variable("NEMOTRON_COMPONENT_SOURCE");
    let out=variable("NEMOTRON_COMPONENT_PILE");
    fs::create_dir(&out)?;
    let path=out.join("components.pile");
    fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
    let key=SigningKey::generate(&mut rand::rngs::OsRng);
    let mut pile=Pile::open(&path)?;
    let policy=CollectionPolicy::new(AdmissionPolicy::direct(key.verifying_key()),AdmissionPolicy::direct(key.verifying_key()));
    let collection: model_collection::ModelCollection=pile.collection("nemotron-component-fixture",policy)?;
    model_collection::require_model_writer(&mut pile,&key,collection)?;
    let mut layers=Fragment::empty();
    let mut records=Vec::new(); // Ephemeral finite ingress receipt, not a graph catalogue.
    for (index,file,size,expected_sha) in SHARDS {
        let handle=fs::File::open(source.join(file))?;
        let before=handle.metadata()?;
        ensure!(before.len()==size,"source size changed");
        let map=unsafe { memmap2::Mmap::map(&handle) }?;
        ensure!(sha256(&map)==expected_sha,"source SHA256 changed");
        let tensors=SafeTensors::deserialize(&map)?;
        let mut members=Fragment::empty();
        let mut read = |suffix: &str, dtype: SourceDtype, shape: &[usize]| -> AnyResult<Vec<u8>> {
            let name=format!("language_model.backbone.layers.{index}.{suffix}");
            let tensor=tensors.tensor(&name)?;
            ensure!(tensor.dtype()==dtype && tensor.shape()==shape,"source dtype/shape mismatch: {name}");
            records.push(json!({"name":name,"dtype":format!("{dtype:?}"),"shape":shape,
                "bytes":tensor.data().len(),"blake3":blake3::hash(tensor.data()).to_hex().to_string()}));
            Ok(tensor.data().to_vec())
        };
        let norm=read("norm.weight",SourceDtype::BF16,&[4096])?;
        members+=native::store_bf16(&mut pile,"norm",[4096],Bytes::from_source(norm))?;
        let operations: &[(&str,[usize;2])] = if index==2 {
            &[("in_proj",[18560,4096]),("out_proj",[4096,8192])]
        } else { &[("q_proj",[4096,4096]),("k_proj",[256,4096]),("v_proj",[256,4096]),("o_proj",[4096,4096])] };
        for &(role,shape) in operations {
            let weight=read(&format!("mixer.{role}.weight"),SourceDtype::F8_E4M3,&shape)?;
            let input=read(&format!("mixer.{role}.input_scale"),SourceDtype::F32,&[1])?;
            let scale=read(&format!("mixer.{role}.weight_scale"),SourceDtype::F32,&[])?;
            let packed=packing::pack_fp8(shape,
                packing::Plane{dtype:packing::Dtype::E4m3Fn,shape:&shape,bytes:&weight},
                packing::Plane{dtype:packing::Dtype::F32,shape:&[],bytes:&scale},
                packing::Plane{dtype:packing::Dtype::F32,shape:&[1],bytes:&input}).map_err(anyhow::Error::msg)?;
            members+=native::store_fp8(&mut pile,role,packed)?;
        }
        if index==2 {
            for (role,n) in [("A_log",128),("D",128),("dt_bias",128),("conv1d.bias",10240),("norm.weight",8192)] {
                let bytes=read(&format!("mixer.{role}"),SourceDtype::BF16,&[n])?;
                members+=native::store_bf16(&mut pile,role,[n as u64],Bytes::from_source(bytes))?;
            }
            let bytes=read("mixer.conv1d.weight",SourceDtype::BF16,&[10240,1,4])?;
            members+=native::store_bf16(&mut pile,"conv1d.weight",[10240,1,4],Bytes::from_source(bytes))?;
        }
        let after=handle.metadata()?;
        ensure!(after.len()==before.len() && after.modified()?==before.modified()?,"source changed while importing");
        layers+=entity! {attrs::kind:native_pile::LAYER,attrs::index:index as u64,attrs::member*:members};
        ensure!(fs::metadata(&path)?.len() < 1<<30,"scratch pile cap");
    }
    let fragment=entity! {attrs::kind:"nemotron-component-fixture",attrs::member*:layers};
    let root=fragment.root().context("fixture root")?;
    model_collection::publish_model_fragment_into(&mut pile,&key,collection,fragment)?;
    pile.close()?;
    ensure!(fs::metadata(&path)?.len()<1<<30,"scratch pile cap");
    let mut reopened=Pile::new(triblespace::core::repo::pile::PileFile::open_read_only(&path)?);
    reopened.refresh()?;
    let store=reopened.snapshot()?;
    let collection=model_collection::open_model_collection(&store,collection.handle())?;
    let snapshot=model_collection::snapshot_model_collection_for(&store,collection)?;
    for index in [2,7] {
        let layer=native_pile::layers(&snapshot,root,index).next().context("reopened layer")?;
        let roles: &[&str]=if index==2 {&["in_proj","out_proj"]} else {&["q_proj","k_proj","v_proj","o_proj"]};
        for role in roles { native_pile::fp8(&snapshot,layer,role).next().context("reopened operation")??; }
        native_pile::auxiliary::<BF16,1,_>(&snapshot,layer,"norm").next().context("reopened norm")??;
    }
    json_write(&out.join("import.json"),&json!({"root":format!("{root:X}"),
        "collection":collection.handle().raw.iter().map(|b|format!("{b:02x}")).collect::<String>(),"support":snapshot.support().len(),
        "kind":"component fixture, NOT a complete model","records":records,
        "source_revision":"da5b2650a835292c678994a1fa3fb653250fb508",
        "source_bytes":145366832,"pile_bytes":fs::metadata(path)?.len(),
        "promotion":"none at ingress; original auxiliary BF16 retained"}));
    Ok(())
}

// Test-only outer operations: no CPU numeric path or new runtime dispatch.
#[cube(launch_unchecked)]
fn outer_norm(x:&Array<bf16>,gain:&Array<bf16>,out:&mut Array<bf16>) {
    let row=CUBE_POS_X as usize;
    let lane=UNIT_POS_X as usize;
    let mut sums=SharedMemory::<f32>::new(64usize);
    let mut sum=0.0f32;
    for j in 0usize..64usize {
        let v=f32::cast_from(x[row*4096usize+lane+j*64usize]);
        sum+=v*v;
    }
    sums[lane]=sum;
    sync_cube();
    let mut step=32usize;
    while step>0usize {
        if lane<step { sums[lane]+=sums[lane+step]; }
        sync_cube(); step/=2usize;
    }
    let inverse=1.0f32/(sums[0usize]/4096.0f32+0.00001f32).sqrt();
    for j in 0usize..64usize {
        let col=lane+j*64usize;
        let rounded=bf16::cast_from(f32::cast_from(x[row*4096usize+col])*inverse);
        out[row*4096usize+col]=bf16::cast_from(f32::cast_from(rounded)*f32::cast_from(gain[col]));
    }
}
#[cube(launch_unchecked)]
fn outer_add(x:&Array<bf16>,y:&Array<bf16>,out:&mut Array<bf16>,len:usize) {
    let i=ABSOLUTE_POS as usize;
    if i<len {out[i]=bf16::cast_from(f32::cast_from(x[i])+f32::cast_from(y[i]));}
}
fn tensor(bytes:&[u8],shape:&[usize],dtype:DType)->CudaTensor {
    let device=CudaDevice{index:0}; let client=CudaRuntime::client(&device);
    CudaTensor::new_contiguous(client.clone(),device,shape.into(),client.create_from_slice(bytes),dtype)
}
fn normalized(x:&CudaTensor,gain:&CudaTensor)->CudaTensor {
    let count=x.meta.shape().as_slice()[1]; let out=x.client.empty(count*4096*2);
    unsafe {outer_norm::launch_unchecked::<CudaRuntime>(&x.client,CubeCount::Static(count as u32,1,1),CubeDim::new_1d(64),
        ArrayArg::from_raw_parts(x.handle.clone(),count*4096),ArrayArg::from_raw_parts(gain.handle.clone(),4096),
        ArrayArg::from_raw_parts(out.clone(),count*4096));}
    CudaTensor::new_contiguous(x.client.clone(),x.device.clone(),[1,count,4096].into(),out,DType::BF16)
}
fn residual(x:&CudaTensor,y:&CudaTensor)->CudaTensor {
    assert_eq!(x.meta.shape(),y.meta.shape());
    let len=x.meta.shape().as_slice().iter().product(); let out=x.client.empty(len*2); let cube=CubeDim::new_1d(64);
    unsafe {outer_add::launch_unchecked::<CudaRuntime>(&x.client,cubecl::calculate_cube_count_elemwise(&x.client,len,cube),cube,
        ArrayArg::from_raw_parts(x.handle.clone(),len),ArrayArg::from_raw_parts(y.handle.clone(),len),ArrayArg::from_raw_parts(out.clone(),len),len);}
    CudaTensor::new_contiguous(x.client.clone(),x.device.clone(),x.meta.shape().clone(),out,DType::BF16)
}
fn raw(t:&CudaTensor)->Vec<u8> {
    let n=t.meta.shape().as_slice().iter().product::<usize>()*if t.dtype==DType::BF16 {2}else{4};
    t.client.read_one(t.handle.clone()).unwrap()[..n].to_vec()
}
fn scalar(view:&TensorView)->f32 {
    assert!(view.dims().is_empty()); f32::from_le_bytes(view.payload()[..].try_into().unwrap())
}
fn aux<const R:usize>(snapshot:&model_collection::ModelSnapshot<impl BlobStoreGet>,layer:Id,role:&str,shape:&[usize],promote:bool)->CudaTensor {
    let view=native_pile::auxiliary::<BF16,R,_>(snapshot,layer,role).next().unwrap().unwrap();
    assert_eq!(view.dims(),shape.iter().map(|&x|x as u64).collect::<Vec<_>>());
    // Typed BF16 query AFTER ingress checked the safetensors header dtype.
    let bytes=if promote {view.payload().chunks_exact(2).flat_map(|b|bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32().to_le_bytes()).collect()}
        else {view.payload().to_vec()};
    tensor(&bytes,shape,if promote {DType::F32}else{DType::BF16})
}
fn projection(view:&native_pile::Fp8Projection)->Projection {
    let dims=view.weight.dims();
    Projection::upload(CudaDevice{index:0},[dims[0] as usize,dims[1] as usize],view.weight.payload(),scalar(&view.input_scale),scalar(&view.weight_scale),128<<20).unwrap()
}

struct Evidence {out:PathBuf,reference:PathBuf,manifest:Value,records:Vec<Value>,metrics:Vec<Value>,used:usize}
impl Evidence {
    fn expected(&self,name:&str,shape:&[usize],dtype:&str)->Vec<u8> {
        let path=format!("{name}.{dtype}");
        let entry=self.manifest["tensors"].as_array().unwrap().iter().find(|r|r["path"]==path).unwrap();
        assert_eq!(entry["shape"],json!(shape)); assert_eq!(entry["dtype"],dtype);
        let size=entry["bytes"].as_u64().unwrap() as usize; assert!(size<=CAP);
        let mut bytes=Vec::new(); fs::File::open(self.reference.join(&path)).unwrap().take(size as u64+1).read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len(),size); assert_eq!(sha256(&bytes),entry["sha256"].as_str().unwrap()); bytes
    }
    fn save(&mut self,name:&str,shape:&[usize],dtype:&str,bytes:&[u8]) {
        self.used=self.used.checked_add(bytes.len()).unwrap(); assert!(self.used<=CAP && self.records.len()<512);
        let path=format!("{name}.{dtype}"); let full=self.out.join(&path);
        fs::create_dir_all(full.parent().unwrap()).unwrap(); assert!(!full.exists()); fs::write(&full,bytes).unwrap();
        self.records.push(json!({"path":path,"shape":shape,"dtype":dtype,"bytes":bytes.len(),"sha256":sha256(bytes)}));
        self.flush();
    }
    fn flush(&self) {json_write(&self.out.join("report.json"),&json!({"payload_bytes":self.used,"tensors":self.records,"metrics":self.metrics,
        "limits":"descriptive diagnostic bands, not model acceptance; stage errors may include upstream differences"}));}
    fn compare(&mut self,name:&str,t:&CudaTensor,reference:&str) {
        let dtype=if t.dtype==DType::BF16 {"bf16"}else{"f32"};
        let shape=t.meta.shape().as_slice(); let actual=raw(t); let expected=self.expected(reference,shape,dtype);
        self.save(name,shape,dtype,&actual); self.metric(name,&actual,&expected,dtype);
    }
    fn metric(&mut self,name:&str,a:&[u8],b:&[u8],dtype:&str) {
        assert_eq!(a.len(),b.len());
        let width=if dtype=="bf16" {2}else{4};
        let decode=|v:&[u8]| if width==2 {bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f64()}else{f32::from_le_bytes(v.try_into().unwrap()) as f64};
        let (atol,rtol)=if name.contains("conv_tail") {(0.,0.)}else if name.contains("state")||name.ends_with("/norm") {(0.002,0.02)}else{(0.02,0.02)};
        let (mut max_abs,mut max_rel,mut outside,mut nonfinite)=(0f64,0f64,0usize,0usize);
        for (a,b) in a.chunks_exact(width).zip(b.chunks_exact(width)) {
            let (x,y)=(decode(a),decode(b));
            if !x.is_finite()||!y.is_finite() {nonfinite+=1;continue;}
            let e=(x-y).abs(); max_abs=max_abs.max(e);max_rel=max_rel.max(e/y.abs().max(1e-6));
            outside+=usize::from(e>atol+rtol*y.abs());
        }
        self.metrics.push(json!({"name":name,"max_abs":max_abs,"max_rel_floor_1e_6":max_rel,"outside_band":outside,"nonfinite":nonfinite,"atol":atol,"rtol":rtol}));
        self.flush(); assert_eq!(nonfinite,0,"nonfinite retained; no retry");
    }
}

fn projection_diagnostic(e:&mut Evidence,name:&str,p:&Projection) {
    let [n,k]=p.shape; let prefix=format!("diagnostic/{name}");
    let input=e.expected(&format!("{prefix}/input"),&[1,3,k],"bf16");
    let q=e.expected(&format!("{prefix}/qbytes"),&[3,k],"e4m3fn");
    let expected=e.expected(&format!("{prefix}/output"),&[1,3,n],"bf16");
    let x=tensor(&input,&[1,3,k],DType::BF16); let client=&p.client; let cube=CubeDim::new_1d(THREADS);
    let native_q=client.empty(3*k);
    unsafe {quantize::launch_unchecked::<CudaRuntime>(client,cubecl::calculate_cube_count_elemwise(client,3*k,cube),cube,
        ArrayArg::from_raw_parts(x.handle.clone(),3*k),ArrayArg::from_raw_parts(native_q.clone(),3*k),3*k,p.input_scale);}
    let native_q=client.read_one(native_q).unwrap();
    e.save(&format!("{prefix}/native_qbytes"),&[3,k],"e4m3fn",&native_q[..3*k]);
    e.metrics.push(json!({"name":prefix,"quantizer_byte_mismatches":native_q[..3*k].iter().zip(&q).filter(|(a,b)|a!=b).count()}));
    let acc=client.empty(3*n*4);let same=client.empty(3*n*2);let stock_q=client.create_from_slice(&q);
    launch_mma(client,&stock_q,&p.weight,&acc,3,n,k).unwrap();
    unsafe {finish::launch_unchecked::<CudaRuntime>(client,cubecl::calculate_cube_count_elemwise(client,3*n,cube),cube,
        ArrayArg::from_raw_parts(acc,3*n),ArrayArg::from_raw_parts(same.clone(),3*n),3*n,p.alpha);}
    let same=client.read_one(same).unwrap();
    e.save(&format!("{prefix}/same_q_output"),&[1,3,n],"bf16",&same[..3*n*2]);
    e.metric(&format!("{prefix}/same_q_output"),&same[..3*n*2],&expected,"bf16");
    let full=p.project(&x,WORK).unwrap();
    e.compare(&format!("{prefix}/native_projection"),&full,&format!("{prefix}/output"));
}

#[test]
#[ignore = "reserved CUDA: explicit reopened native component pile and frozen stock reference"]
fn actual_components_gpu() -> AnyResult<()> {
    let pile_dir=variable("NEMOTRON_COMPONENT_PILE");
    let imported:Value=serde_json::from_slice(&fs::read(pile_dir.join("import.json"))?)?;
    let root=Id::from_hex(imported["root"].as_str().unwrap()).unwrap();
    let collection=model_collection::parse_collection_handle(imported["collection"].as_str().unwrap())?;
    let mut pile=Pile::new(triblespace::core::repo::pile::PileFile::open_read_only(&pile_dir.join("components.pile"))?);
    pile.refresh()?; let store=pile.snapshot()?;
    let collection=model_collection::open_model_collection(&store,collection)?;
    let snapshot=model_collection::snapshot_model_collection_for(&store,collection)?;
    let reference=variable("NEMOTRON_COMPONENT_REFERENCE");
    let out=variable("NEMOTRON_COMPONENT_OUT"); fs::create_dir(&out)?;
    let manifest_raw=fs::read(reference.join("tensors.json"))?;
    ensure!(manifest_raw.len()<1<<20,"reference manifest cap");
    ensure!(sha256(&manifest_raw)==std::env::var("NEMOTRON_COMPONENT_MANIFEST_SHA256")?,"reference manifest pin");
    let manifest:Value=serde_json::from_slice(&manifest_raw)?;
    ensure!(manifest["payload_bytes"].as_u64().unwrap()<=CAP as u64,"reference cap");
    let mut e=Evidence{out,reference,manifest,records:Vec::new(),metrics:Vec::new(),used:0};
    let residual_bytes=e.expected("residual",&[1,3,4096],"bf16");
    let layer=native_pile::layers(&snapshot,root,2).next().context("layer2")?;
    let gain=aux::<1>(&snapshot,layer,"norm",&[4096],false);
    let ip=projection(&native_pile::fp8(&snapshot,layer,"in_proj").next().unwrap()?);
    let op=projection(&native_pile::fp8(&snapshot,layer,"out_proj").next().unwrap()?);
    projection_diagnostic(&mut e,"mamba.in_proj",&ip);projection_diagnostic(&mut e,"mamba.out_proj",&op);
    let conv=aux::<3>(&snapshot,layer,"conv1d.weight",&[10240,1,4],false);
    let bias=aux::<1>(&snapshot,layer,"conv1d.bias",&[10240],false);
    let a=aux::<1>(&snapshot,layer,"A_log",&[128],true);let d=aux::<1>(&snapshot,layer,"D",&[128],true);
    let dt=aux::<1>(&snapshot,layer,"dt_bias",&[128],true);let mg=aux::<1>(&snapshot,layer,"norm.weight",&[8192],false);
    let params=mamba2::gpu::Parameters{conv_weight:&conv,conv_bias:Some(&bias),a_log:&a,dt_bias:&dt,d:&d,norm_gain:&mg};
    let config=mamba2::contract::Config{heads:128,head_dim:64,groups:8,state_dim:128,conv_kernel:4,norm_epsilon:1e-5,time_step_min:0.0};
    let mut previous:Option<mamba2::gpu::State<CudaRuntime>>=None;
    for (arm,counts) in PLANS {
        let mut state=if arm=="reset" {Some(previous.take().unwrap().reset(WORK).map_err(anyhow::Error::msg)?)}else{None};
        let mut start=0;let mut mixers=Vec::new();let mut sums=Vec::new();
        for (part,&count) in counts.iter().enumerate() {
            let x=tensor(&residual_bytes[start*4096*2..(start+count)*4096*2],&[1,count,4096],DType::BF16);
            let norm=normalized(&x,&gain);
            if arm=="T3" {e.compare("mamba/norm",&norm,"mamba/norm");}
            let projected=ip.project(&norm,WORK).map_err(anyhow::Error::msg)?;
            let got=mamba2::gpu::chunk(config,&params,&projected,state.as_ref(),WORK).map_err(anyhow::Error::msg)?;
            let kind=if arm=="steps" || (arm=="split" && part==1) {"step"}else{"prefill"};
            let prefix=format!("mamba/{arm}/part{part}-{kind}");
            e.compare(&format!("{prefix}/projected"),&projected,&format!("{prefix}/projected"));
            e.compare(&format!("{prefix}/output"),&got.values,&format!("{prefix}/output"));
            let mixer=op.project(&got.values,WORK).map_err(anyhow::Error::msg)?;
            mixers.extend(raw(&mixer));sums.extend(raw(&residual(&x,&mixer)));
            state=Some(got.state);start+=count;
            if part+1==counts.len() {
                let s=state.as_ref().unwrap();assert_eq!(s.attended(),start);
                e.compare(&format!("{prefix}/state"),s.recurrent(),&format!("{prefix}/state"));
                let history=raw(s.conv());
                let tail:Vec<u8>=history.chunks_exact(8).flat_map(|row|row[2..].iter().copied()).collect();
                let expected=e.expected(&format!("{prefix}/conv_tail"),&[1,10240,3],"bf16");
                e.save(&format!("{prefix}/conv_tail"),&[1,10240,3],"bf16",&tail);
                e.metric(&format!("{prefix}/conv_tail"),&tail,&expected,"bf16");
            }
        }
        for (label,bytes) in [("mixer",mixers),("residual",sums)] {
            let name=format!("mamba/{arm}/{label}");let expected=e.expected(&name,&[1,start,4096],"bf16");
            e.save(&name,&[1,start,4096],"bf16",&bytes);e.metric(&name,&bytes,&expected,"bf16");
        }
        previous=state;
    }
    drop((ip,op,previous));
    let layer=native_pile::layers(&snapshot,root,7).next().context("layer7")?;
    let gain=aux::<1>(&snapshot,layer,"norm",&[4096],false);
    let views=["q_proj","k_proj","v_proj","o_proj"].map(|role|native_pile::fp8(&snapshot,layer,role).next().unwrap().unwrap());
    for (role,view) in ["q_proj","k_proj","v_proj","o_proj"].iter().zip(&views) {projection_diagnostic(&mut e,&format!("attention.{role}"),&projection(view));}
    let operations=views.each_ref().map(|v|attention::Operation{e4m3fn:v.weight.payload(),input_scale:scalar(&v.input_scale),weight_scale:scalar(&v.weight_scale)});
    let mut attn=attention::Attention::upload(CudaDevice{index:0},attention::SourceBinding{model_root:root,layer,operations:views.each_ref().map(|v|v.operation)},operations,
        attention::Limits{context_limit:3,capacity:3,max_resident_bytes:64<<20,max_working_bytes:WORK}).map_err(anyhow::Error::msg)?;
    for (arm,counts) in PLANS {
        attn.reset().map_err(anyhow::Error::msg)?;
        let mut start=0;
        for (part,&count) in counts.iter().enumerate() {
            let x=tensor(&residual_bytes[start*4096*2..(start+count)*4096*2],&[1,count,4096],DType::BF16);
            let norm=normalized(&x,&gain);if arm=="T3" {e.compare("attention/norm",&norm,"attention/norm");}
            let got=attn.enqueue(&norm,start).map_err(anyhow::Error::msg)?;
            attn.commit().map_err(anyhow::Error::msg)?;
            let prefix=format!("attention/{arm}/part{part}");
            for (name,t) in [("q",&got.stages.q),("k",&got.stages.k),("v",&got.stages.v),("attention",&got.stages.attention),("mixer",&got.values)] {
                e.compare(&format!("{prefix}/{name}"),t,&format!("{prefix}/{name}"));
            }
            e.compare(&format!("{prefix}/residual"),&residual(&x,&got.values),&format!("{prefix}/residual"));start+=count;
        }
        let [keys,values]=attn.diagnostic_kv().map_err(anyhow::Error::msg)?;
        for (name,bytes) in [("keys",keys),("values",values)] {
            let prefix=format!("attention/{arm}/{name}");let expected=e.expected(&prefix,&[1,start,256],"bf16");
            e.save(&prefix,&[1,start,256],"bf16",&bytes);e.metric(&prefix,&bytes,&expected,"bf16");
        }
    }
    e.flush(); Ok(())
}
