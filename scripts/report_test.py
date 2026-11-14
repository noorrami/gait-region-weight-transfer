"""Render prespecified test summaries from locked artifacts; no new model analysis."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.evaluation import digest


def main():
    output=ROOT/'results/test/v1'
    s=json.loads((output/'summary.json').read_text())
    verified=json.loads((output/'verification.json').read_text())
    assert verified['status']=='verified'
    figs=output/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.dpi':300})
    methods=['full','equal','bag_selected','normal_selected','upper_only','lower_only']
    labels=['Full','Equal','BG-weight','NM-weight','Upper','Lower']
    colors=['#9CA3AF','#303030','#0078BF','#8464A5','#DB8D2D','#2D8C72']
    primary=s['rank1']['pca_then_regularized_lda']
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),sharey=True)
    for ax,condition,title in zip(axes,['nm','bg','cl'],['Normal walking (NM)','Bag carrying (BG)','Coat wearing (CL)']):
        values=[primary[condition][m]['mean_rank1']*100 for m in methods]
        ax.bar(np.arange(6),values,color=colors,width=.72)
        for x,y in enumerate(values):ax.text(x,y+1.5,f'{y:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(np.arange(6),labels,rotation=35,ha='right');ax.set_ylim(0,110)
        ax.set_yticks([0,20,40,60,80,100]);ax.set_title(title,fontsize=11,color='#0078BF')
        ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Rank-1 identification (%)')
    fig.suptitle('Held-out test: prespecified PCA + regularized LDA backbone',fontsize=12)
    fig.text(.5,.015,'47 people; equal mean over 54 and 90 degree same-view evaluations; all models and weights fixed before test.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    for suffix in ('png','svg'):fig.savefig(figs/f'lda_test_rank1.{suffix}')
    plt.close(fig)
    c=s['contrasts'][s['primary_contrast']];lo,hi=c['ci95_percentile_pp'];estimate=c['difference_pp']
    fig,ax=plt.subplots(figsize=(8.4,2.8))
    ax.axvline(0,color='#555555',linestyle='--',linewidth=1)
    ax.hlines(0,lo,hi,color='#0078BF',linewidth=2)
    ax.plot(estimate,0,'o',color='#0078BF',markersize=8)
    ax.set_yticks([0],['Coat: BG-selected minus equal'])
    ax.set_ylim(-.7,.7);ax.set_xlim(min(lo-1,-1),max(hi+1,1))
    ax.set_xlabel('Paired Rank-1 difference (percentage points; positive favors BG-selected)')
    ax.set_title('Prespecified primary transfer contrast',color='#0078BF',fontsize=12)
    ax.text(estimate,.23,f'{estimate:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f})',ha='center',fontsize=10)
    fig.text(.5,.025,'Paired bootstrap: 47 probe people, 10,000 draws; gallery and trained models fixed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    for suffix in ('png','svg'):fig.savefig(figs/f'primary_transfer_difference.{suffix}')
    plt.close(fig)
    names={'full':'الجسم الكامل','equal':'الدمج المتساوي','bag_selected':'الوزن المختار من الحقيبة',
           'normal_selected':'الوزن المختار من المشي العادي','upper_only':'الجزء العلوي فقط','lower_only':'الجزء السفلي فقط'}
    text=['# نتيجة الاختبار المحجوز الأولى','',
          '12 نوفمبر 2026. أُجري التقييم على 47 شخصًا لم يدخلوا تعلّم الإسقاط أو اختيار المعاملات. استُخدمت النماذج والتطبيع والأوزان المقفلة دون إعادة تعلّم أو ضبط. نجح تجهيز 940 ملف GEI من 70,837 إطارًا، دون استبعاد إضافي. لكل حالة 188 مقطع probe: تكراران لكل حالة عند كل زاوية لكل شخص (47×2×2)، مع أربع صور GEI للمشي العادي لتسجيل كل شخص عند كل زاوية.',
          '', '## جواب سؤال البحث الأساسي','',
          f"**لم تدعم التجربة فرضية تحسن التحقق من الهوية بالمشية عبر نقل أوزان الحقيبة إلى المعطف.** في النموذج الأساسي PCA+LDA المنظم، انخفض Rank-1 للمعطف من {100*primary['cl']['equal']['mean_rank1']:.2f}% بالدمج المتساوي إلى {100*primary['cl']['bag_selected']['mean_rank1']:.2f}% بوزن الحقيبة المقفل. الفرق {estimate:+.2f} نقطة مئوية، وفاصل الثقة 95% [{lo:+.2f}، {hi:+.2f}]. الفاصل كله سالب في هذا التقييم المشروط بالمعرض والنماذج الثابتة؛ لا يمثل احتمالية أن الفرضية صحيحة ولا تعميمًا على جميع قواعد المشية.",
          '', 'المقياس المحسوب هو **التعرّف المغلق Rank-1 (identification)**، لا EER للتحقق الثنائي (verification). يُحافَظ على هذا التمييز في عنوان/صياغة المخطوط.',
          '', '## النتائج الكاملة','',
          'الأرقام نسب مئوية. المتوسط متساوٍ بين الزاويتين؛ الوحدة الإحصائية الشخص وليست الإطار أو مقارنة gallery pair. مقاطع المقارنة واحدة بين الطرق.',
          '', '| النموذج | الحالة | الطريقة | 54° | 90° | المتوسط |','|---|---|---|---:|---:|---:|']
    for backbone in ['pca_then_regularized_lda','pca']:
        for condition in ['nm','bg','cl']:
            for method in methods:
                r=s['rank1'][backbone][condition][method]
                text.append(f"| {'PCA+LDA' if backbone!='pca' else 'PCA'} | {condition.upper()} | {names[method]} | {r['per_view_rank1'][0]*100:.2f} | {r['per_view_rank1'][1]*100:.2f} | {r['mean_rank1']*100:.2f} |")
    text+=['','## المقارنات المقترنة للمعطف','',
           'كل سطر: وزن الحقيبة المقفل ناقص المقارنة. السطر الأساسي محدد مسبقًا؛ باقي الفواصل وصفية غير مصححة لتعدد المقارنات. لا نختار النموذج الأعلى بعد رؤية الاختبار.',
           '', '| النموذج | المقارنة | الفرق بالنقاط المئوية | فاصل 95% | الدور |', '|---|---|---:|---|---|']
    for backbone in ['pca_then_regularized_lda','pca']:
        for control in ['equal','full','upper_only','lower_only','normal_selected']:
            v=s['contrasts'][f'{backbone}_cl_bag_selected_minus_{control}'];a,b=v['ci95_percentile_pp']
            text.append(f"| {'PCA+LDA' if backbone!='pca' else 'PCA'} | {names[control]} | {v['difference_pp']:+.2f} | [{a:+.2f}، {b:+.2f}] | {'أساسي' if v['primary'] else 'وصفي ثانوي'} |")
    text+=['','## تفسير محدود بالأدلة','',
           '1. تحسن LDA في مجموعة اختيار الحقيبة (54.35% إلى 63.04%) لم يتكرر على أشخاص اختبار الحقيبة: الدمج المتساوي 65.43% مقابل 60.11% للوزن المختار. لذلك لا ننسب الإخفاق إلى الانتقال للمعطف وحده؛ تعميم الوزن على هويات جديدة في حالة المصدر نفسها لم يتحسن أيضًا. الإفراط في الاختيار/اختلاف الأشخاص تفسيران محتملان، لا آلية سببية مثبتة.',
           '2. على المعطف، الجزء السفلي وحده بلغ 31.91% في LDA مقابل 15.96% للوزن المختار. هذه مقارنة ضابطة مقررة سلفًا، وتمنع الادعاء بأن الدمج المقترح تفوق على الأجزاء الفردية. وزن الحقيبة كان 0.8 للعلوي و0.2 للسفلي بعد التطبيع؛ لا نعيد اختياره الآن من المعطف.',
           '3. دقة المشي العادي في LDA مرتفعة (99.47% للدمج المتساوي)، بينما ظروف الحقيبة والمعطف أصعب في هذا البروتوكول. هذا نمط وصفي ولا يثبت أن كل مراحل المعالجة خالية من القصور.',
           '4. PCA ثانوي بقي دون تحسن للمعطف: المتساوي 14.36% ووزن الحقيبة 13.83%. لا نعكس ترتيب النموذج الأساسي أو نختار طريقة جديدة استجابة للاختبار.',
           '', '## حدود الاستدلال','',
           'إعادة أخذ العينات تستخدم 47 شخص probe مع إبقاء زاويتي الشخص وتكراريه معًا، 10,000 مرة بالبذرة 20261111، ومئينات 2.5/97.5 باستيفاء خطي. المعرض والنماذج ثابتة؛ لا يشمل الفاصل اختلاف تعلّم النماذج أو اختيار معرض جديد أو تغيّر قاعدة البيانات. النتائج تخص النسخة المعالجة والتقسيم والمنهج الحالي؛ لا تثبت فشل كل نقل أوزان ممكن.',
           '',
           'نسخة Kaggle سابقة المعالجة، ومسار معالجتها والتطابق مع الأصل والترخيص لدى الرافع غير موثقين بالكامل. نصفا GEI تقسيم هندسي تقريبي، وحد 20 إطارًا ليس ضمان دورة مشي. الزاويتان ثابتتان، والبيئة واحدة، وعدد الأشخاص محدود. لا تنسب النتيجة السلبية تلقائيًا إلى المعالجة السابقة؛ لا توجد مقارنة بأصل موثّق تثبت ذلك. الجدة والنشر غير مضمونين، والمراجعة الأدبية محدودة.',
           '', '## التحقق والآثار','',
           'نجحت 42 تجربة تحقق برمجية قبل فتح الاختبار. أعيد حساب 6,768 تنبؤًا و72 مقارنة حسب الزاوية من مصفوفات الدرجات المحفوظة، وأعيد إنتاج 14 توزيع bootstrap وفواصلها؛ طابقت السجلات دون تعادلات قمة دقيقة. جرى التحقق من بصمات الخطة والكود والفهارس والنماذج والمخرجات. هذه إعادة تحقق من الحسابات وليست تشغيلًا لاختيار نتيجة أخرى.',
           '',
           'المراجع الداخلية: configs/study.frozen.json؛ configs/freeze_manifest.json؛ configs/evaluation.v2.json؛ results/development/v1/selection.lock.json؛ results/test/v1/{summary.json,predictions.csv,scores.npz,correctness.npz,bootstrap.npz,result_manifest.json,verification.json}. تجهيز GEI موثق في data/processed/kaggle_casia_b_v1/test_gei_v1/provenance.json.',
           '',
           'حُفظت مواصفات التقييم v1، ثم صُحح قبل حساب أول درجة اختبار حقل اسم المقارنة في CSV ضمن v2؛ المصدر القديم محفوظ. هذا تصحيح تتبع لا يغيّر الدرجات أو العينات أو الاستدلال. بعد النتيجة لم تتغير النماذج أو الأوزان أو الاستبعادات.',
           '', '## الرسوم','',
           'results/test/v1/figures/lda_test_rank1.png ونسخته SVG: نتائج النموذج الأساسي في الحالات الثلاث، بالمقارنات الست المقررة. results/test/v1/figures/primary_transfer_difference.png وSVG: فرق المعطف الأساسي وفاصل الثقة؛ يمين الصفر يفضّل وزن الحقيبة.',
           '',
           'الرسوم مستخرجة من summary.json المقفل، باللون الأزرق المرجعي #0078BF. خط الرسم Arial بحجم 10 غالبًا؛ خطا Helvetica LT Std وCharis SIL في ورقة الدكتور غير مثبتين محليًا، لذا هذا بديل معلن للرسوم، وليس ادعاء مطابقة قالب المخطوط النهائي. الصور 300 dpi وSVG قابل للتكبير.',
           '', '## المتابعة العلمية','',
           'نحتفظ بالنتيجة السلبية ونبدأ صياغة تقرير/مناقشة صادقة لسؤال النقل، مع إكمال مراجعة السوابق قبل ادعاء قيمة نشر جديدة. أي تعديل لاحق للمنهج أصبح تطويرًا بعد الاطلاع على الاختبار؛ يجب تسجيله كتجربة استكشافية جديدة، ولا يمكن استخدام الأشخاص أنفسهم مرة أخرى كاختبار نهائي غير مرئي.', '']
    # Use identification consistently; do not describe Rank-1 as binary verification.
    text=[line.replace('تحسن التحقق من الهوية بالمشية','تحسن التعرّف على الهوية بالمشية') for line in text]
    Path(ROOT/'docs/final_test_result_ar.md').write_text('\n'.join(text),encoding='utf-8')
    artifacts=[str(p.relative_to(ROOT)).replace('\\','/') for p in sorted(figs.glob('*'))]+['docs/final_test_result_ar.md']
    (output/'report_manifest.json').write_text(json.dumps({'source_summary_sha256':digest(output/'summary.json'),
        'script_sha256':digest(Path(__file__)),'artifacts':{p:digest(ROOT/p) for p in artifacts}},indent=2)+'\n',encoding='utf-8')
    print('Arabic result report and four PNG/SVG figure artifacts generated.')


if __name__=='__main__':main()
