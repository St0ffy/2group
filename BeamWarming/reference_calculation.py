import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, json
from pathlib import Path
OUT=Path(__file__).resolve().parent; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
N0=200
METHODS=['BW','Upwind','LW','minmod','vanLeer','superbee','MC']
R_VALUES=[0.5,0.9]
PROFILES=['step','rectangle','gaussian','sinusoid']
NAMES={'step':'Ступенька','rectangle':'Прямоугольник','gaussian':'Гауссиана','sinusoid':'Синусоида','BW':'Бим--Уорминг','Upwind':'Противопоточная','LW':'Лакс--Вендрофф','minmod':'minmod','vanLeer':'van Leer','superbee':'superbee','MC':'MC'}

def initial(x, profile):
    if profile=='step': return np.where(x>=0.5,1.,0.)
    if profile=='rectangle': return np.where((x>=0.3)&(x<=0.6),1.,0.)
    if profile=='gaussian': return np.exp(-(x-0.5)**2/(2*0.05**2))
    if profile=='sinusoid': return np.sin(2*np.pi*x)

def phi(theta,method):
    if method=='Upwind':return np.zeros_like(theta)
    if method=='LW':return np.ones_like(theta)
    if method=='minmod':return np.maximum(0,np.minimum(1,theta))
    if method=='vanLeer':return (theta+np.abs(theta))/(1+np.abs(theta))
    if method=='superbee':return np.maximum(np.maximum(np.maximum(np.zeros_like(theta),np.minimum(2*theta,1)),np.minimum(theta,2)),0)
    if method=='MC':return np.maximum(0,np.minimum(np.minimum(2*theta,(1+theta)/2),2))
    raise ValueError(method)

def flux(y, a, h, dt, method):
    v=abs(a)*dt/h
    if a==0: return np.zeros_like(y)
    d=np.roll(y,-1)-y
    if a>0:
        up=a*y
        numerator=y-np.roll(y,1)
    else:
        up=a*np.roll(y,-1)
        numerator=np.roll(y,-2)-np.roll(y,-1)
    theta=np.zeros_like(y)
    np.divide(numerator,d,out=theta,where=(d!=0))
    return up+abs(a)/2*(1-v)*phi(theta,method)*d

def step(y,a,h,dt,method):
    if method=='BW':
        r=abs(a)*dt/h
        if a>0:
            return y-r*(y-np.roll(y,1))-r*(1-r)/2*(y-2*np.roll(y,1)+np.roll(y,2))
        elif a<0:
            return y+r*(np.roll(y,-1)-y)-r*(1-r)/2*(np.roll(y,-2)-2*np.roll(y,-1)+y)
        return y.copy()
    f=flux(y,a,h,dt,method)
    return y-dt/h*(f-np.roll(f,1))

def TV(y):return np.abs(np.roll(y,-1)-y).sum()

def metrics(y, exact,h):
    return {'L1':float(h*np.abs(y-exact).sum()),'C':float(np.max(np.abs(y-exact))),
            'min':float(y.min()), 'max':float(y.max()), 'TV':float(TV(y))}

def simulate(N,r,profile,method,a=1.):
    h=1/N;x=np.arange(N)*h;y=initial(x,profile)
    exact=y.copy() if a in [-1.,1.] else initial(np.mod(x-a,1),profile)
    t=0.; max_tv_inc=0.; cons_error=0.; initmass=float(y.sum()*h); max_eq=0.
    n=0
    while t<1.-1.e-13:
        dt=min(r*h/abs(a),1.-t) if a!=0 else 1.-t
        yy=step(y,a,h,dt,method)
        tv_inc=TV(yy)-TV(y)
        max_tv_inc=max(max_tv_inc,tv_inc)
        cons_error=max(cons_error,abs(yy.sum()*h-initmass))
        if method=='BW' and a>0:
            ff=a*(y+(1-a*dt/h)/2*(y-np.roll(y,1)))
            yf=y-dt/h*(ff-np.roll(ff,1))
            max_eq=max(max_eq,float(np.max(np.abs(yf-yy))))
        y=yy;t+=dt;n+=1
        if n>100000:raise RuntimeError('loop')
    return y,metrics(y,exact,h),{'max_tv_inc':float(max_tv_inc),'mass_error':float(cons_error),'equivalence_error':float(max_eq),'steps':n}

rows=[]; diagnostics=[]
for profile in PROFILES:
 for r in R_VALUES:
  for meth in METHODS:
   y,m,checks=simulate(N0,r,profile,meth)
   rows.append({'profile':profile,'r':r,'method':meth,**m,**checks})
   diagnostics.append(checks)
  # graphs 7 lines + exact; two-panel separate for readability with explicit exact
  xx=np.arange(N0)/N0
  fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
  for ax,ms in zip(axs,[['BW','Upwind','LW'],['minmod','vanLeer','superbee','MC']]):
   ax.plot(xx,initial(xx,profile),'k--',lw=1.8,label='Точное')
   for meth in ms:
    y,_,_=simulate(N0,r,profile,meth)
    ax.plot(xx,y,lw=1.15,label=NAMES[meth].replace('--','–'))
   ax.set(xlabel='x',ylabel='u',title='Схемы без ограничителей' if ms[0]=='BW' else 'Схемы с ограничителями')
   ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.suptitle(f'{NAMES[profile]}, N={N0}, |r|={r}, t=1')
  fig.tight_layout(); fig.savefig(FIG/f'{profile}_r{str(r).replace(".","")}.pdf',bbox_inches='tight');plt.close(fig)

with open(OUT/'all_results.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
order=[]
for meth in ['BW','minmod','vanLeer']:
 prev=None
 for N in [50,100,200,400,800]:
  y,m,ch=simulate(N,0.5,'sinusoid',meth)
  p1=np.log2(prev['L1']/m['L1']) if prev else None
  pc=np.log2(prev['C']/m['C']) if prev else None
  order.append({'method':meth,'N':N,'L1':m['L1'],'C':m['C'],'pL1':p1,'pC':pc})
  prev=m
with open(OUT/'order_results.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(order[0]));w.writeheader();w.writerows(order)
for meth in ['BW','minmod','vanLeer']:
 ss=[s for s in order if s['method']==meth]
 fig,ax=plt.subplots(figsize=(6,4))
 ax.loglog([x['N'] for x in ss],[x['L1'] for x in ss],'o-',label='L1')
 ax.loglog([x['N'] for x in ss],[x['C'] for x in ss],'s-',label='C')
 ax.set(xlabel='N',ylabel='Ошибка',title=f'Синусоида: {meth}, |r|=0.5')
 ax.grid(alpha=.3,which='both');ax.legend();fig.tight_layout();fig.savefig(FIG/f'order_{meth}.pdf');plt.close(fig)
neg_checks=[]
for meth in METHODS:
 _,m,ch=simulate(200,0.5,'rectangle',meth,a=-1.)
 neg_checks.append({'method':meth,**m,**ch})
with open(OUT/'negative_velocity.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(neg_checks[0])); w.writeheader();w.writerows(neg_checks)
print('results=',len(rows),'max cons=',max(s['mass_error'] for s in rows),'max direct=',max(s['equivalence_error'] for s in rows))
print('TVD step-by-step check (YES = no increase over 1e-12):')
for r in R_VALUES:
 for m in METHODS:
  rr=[row for row in rows if row['r']==r and row['method']==m]
  print(r,m,all(x['max_tv_inc'] <= 1e-12 for x in rr))
print('orders')
for rec in order: print(rec)
print('neg',neg_checks)
