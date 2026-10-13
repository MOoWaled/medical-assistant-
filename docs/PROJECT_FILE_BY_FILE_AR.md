# الدليل الشامل لملفات مشروع Medical Condition Assistant

> آخر تدقيق لهذا الدليل: 10 أكتوبر 2026.  المسار الذي جرى فحصه هو جذر مشروع `Medical_ChatBot`.
>
> **حالة التحقق الحالية:** مجلد `model_API/` استُعيد وهو موجود ويحتوي baseline trainer وFlask API وRNN/LSTM pipeline وNotebook وartifacts وcheckpoints. تم فحص syntax لجميع ملفاته، والتحقق من أن Notebook RNN/LSTM (24 cell) صالحة، واختبار `POST /predict` و`POST /predict/deep-learning`؛ جميعها أعادت HTTP 200 لطلب Athlete’s foot. التدريب يمكن تنفيذه من الـ Notebook أو مباشرةً من `deep_learning_pipeline.py`.

## 1) الصورة الكبيرة: كيف تنتقل البيانات في المشروع؟

```text
NHS Inform A–Z
   │
   ├─ scraper/nhs_scraper.py ──────────────► scraper/nhs_conditions.csv
   │                                             │
   ├─ scraper/LLM_scraper.py + Mistral/ngrok ────┤ (ملء حقول ناقصة + مراجعة بشرية)
   │                                             ▼
   │                                    scraper/nhs_conditions_final.csv
   │                                             │
   ├─ scraper/transferToMongodb.ipynb ──► MongoDB (اختياري)
   │                                             │
   └─ preprocessing_API/create_usable_dataset.py ┴► dataset/usable_dataset.csv
                                                        │
               ┌────────────────────────────────────────┴────────────────────────┐
               ▼                                                                 ▼
  model_API/train_baseline.py                               rag_llm_API/build_faiss_index.py
  TF-IDF symptoms + FAISS محلي                              embeddings + semantic FAISS اختياري
               │                                                                 │
               └────────────────────► rag_llm_API/retriever.py ◄───────────────┘
                                              │
                         model_API/app.py (Flask :5000)
                                              │
                                  gui/app.py (Streamlit :8501)
                                              │
                       Baseline / RNN / LSTM / Qwen / Mistral
```

**الفكرة الأمنية/المنهجية الأساسية:** النموذج لا يكتب Symptoms أو Causes أو Warnings أو Recommendations من خياله. يسترجع المشروع سجل حالة من الداتا المحلية، ثم يعرض الحقول المحفوظة فيه. Qwen/Mistral — عند تفعيلهما — من المفترض أن يختارا اسمًا من المرشحين فقط.

## 2) ما معنى المصطلحات في الكود؟

| المصطلح | المعنى داخل المشروع |
| --- | --- |
| `condition` / `Condition_name` | اسم المرض أو الحالة، مثل `Athlete's foot`. |
| `Symptoms` | نص أعراض الحالة من المصدر. وهو حقل البحث الرئيسي في baseline. |
| `Causes`, `Warnings`, `Recommendations` | النصوص الثابتة التي تعرضها الواجهة بعد اختيار الحالة. |
| RAG | Retrieval-Augmented Generation: استرجاع سجلات قبل استخدام LLM. هنا الاسترجاع هو مصدر الحقائق. |
| TF-IDF | تحويل النص إلى متجه أرقام حسب أهمية الكلمات/الـ n-grams. |
| FAISS | مكتبة تبحث بسرعة في أقرب المتجهات. |
| checkpoint | ملف وزن محفوظ لنموذج Deep Learning بعد Epoch. |
| ngrok | نفق Public URL لخادم يعمل في Kaggle/Colab. ليس ضروريًا للـ baseline المحلي. |

---

# 3) ملفات الجذر Root files

## `README.md` — شرح المشروع العام

**المكان:** `README.md`، 165 سطرًا.

- الأسطر 1–13: بيانات التحدي وصاحب المشروع واسمه وحسابه.
- 15–29: تعريف المشروع وقاعدة الـ grounding وخيارات الواجهة. لاحظ أن قائمة النماذج في README قد تكون أقدم من الملفات الموجودة؛ هذا طبيعي إذا عُدلت الواجهة ولم يُحدّث README.
- 31–59: رحلة الداتا: scraping ثم LLM-assisted completion ثم MongoDB ثم preprocessing.
- 61–99: Mermaid architecture و5 قواعد تمنع الـ LLM من اختراع مرض أو facts.
- 101–124: features والتقنيات المستخدمة.
- 126–152: أوامر التشغيل اليدوي والتشغيل بـ `.bat` وإعداد Qwen/Mistral.
- 154–165: tree مختصر، مثال Athlete’s foot، القيود الطبية، الترخيص.

**كيف تقرؤه؟** هو ملف تعريف/عرض للمشروع على GitHub؛ لا يحتوي دوال Python ولا يُنفّذ.

## `requirements.txt` — مكتبات النسخة الكاملة

**المكان:** `requirements.txt`، 23 سطرًا.

كل سطر `package>=version` يقول لـ `pip` أقل نسخة مقبولة:

| الأسطر | الحزمة | دورها |
| --- | --- | --- |
| 2–5 | `beautifulsoup4`, `cloudscraper`, `lxml`, `pymongo` | scraping HTML وMongoDB. |
| 8–10 | `Flask`, `requests`, `streamlit` | APIs، HTTP requests، والواجهة. |
| 13–17 | `joblib`, `nltk`, `numpy`, `pandas`, `scikit-learn` | تخزين artifacts، معالجة نصوص/جداول، TF-IDF. |
| 20–23 | `faiss-cpu`, `langchain-*`, `sentence-transformers`, `torch`, `jupyterlab` | RAG semantic، notebooks وDeep Learning. |

**مثال:** `pip install -r requirements.txt` يثبت هذه الاعتمادات داخل `.venv` عندما يشغّل launcher.

## `run_the_ChatBot.bat` — مشغّل Windows للنسخة الكاملة

**المكان:** `run_the_ChatBot.bat`، 63 سطرًا.

| الأسطر | ما يحدث |
| --- | --- |
| 1–6 | `@echo off` يخفي الأوامر، و`%~dp0` يحصل على مجلد ملف الـ bat، و`cd /d` ينتقل إليه حتى إن كان على Drive مختلف. |
| 8–14 | يضبط `OPENBLAS_NUM_THREADS` و`OMP_NUM_THREADS` وMKL إلى 1. هذا يمنع خطأ OpenBLAS Memory Allocation. |
| 17–29 | يبحث عن `py -3` ثم `python`. إن لم يجد Python 3 ينهي برسالة مفهومة. |
| 32–38 | ينشئ virtual environment في `.venv` إذا لم يكن موجودًا. |
| 41–45 | يحدّث pip ويثبت `requirements.txt`. أي failure يقفز إلى `:install_error`. |
| 47–50 | لا يحمّل NLTK data عند كل تشغيل لأن السيرفر العام قد يعطي HTTP 429؛ preprocessor لديه fallback محلي. |
| 52–54 | يشغّل `model_API/train_baseline.py` كي يبني/يتحقق من TF-IDF + FAISS. |
| 58 | يفتح نافذة جديدة لتشغيل Flask API على 5000. |
| 61 | يفتح نافذة جديدة لتشغيل Streamlit على 8501. |
| 63–71 | رسالة الرابط، `pause`، ثم handler للأخطاء. |

**مثال فعلي للأثر:** لو كان مسار المشروع `D:\Medical_ChatBot\` يصبح `PROJECT_DIR=D:\Medical_ChatBot\`، فلا تعتمد الأوامر على مكان فتح terminal.

## `.gitignore` — ما لا يُرفع إلى Git

**المكان:** `.gitignore`، قائمة كبيرة مأخوذة من Python template.

أهم مجموعات السطور:

- `__pycache__/`, `*.pyc`: bytecode يتولد آليًا.
- `.venv`, `venv/`, `env/`: المكتبات المحلية الضخمة التي يعيد launcher تثبيتها.
- `.env`, `.streamlit/secrets.toml`: مفاتيح وsecrets لا يجب رفعها.
- `.ipynb_checkpoints`: نسخ Jupyter المؤقتة.
- `model_API/checkpoints/`: checkpoints كبيرة قابلة لإعادة التدريب؛ لذلك لا ترفع تلقائيًا.

## `.idea/.name`

ملف إعدادات IntelliJ/PyCharm محلي. ليس جزءًا من المنطق ولا ينبغي أن يؤثر على التشغيل.

## `model_API.zip`

نسخة ZIP احتياطية غير مستخدمة وقت التشغيل. الـ launcher والـ imports يقرآن فقط المجلد المفكوك `model_API/`. لا ترفعه إلى GitHub إن كان يحتوي artifacts/checkpoints مكررة وكبيرة؛ احتفظ به كنسخة استرجاع محلية فقط.

---

# 4) مجلد `scraper/`: جمع البيانات

## `scraper/nhs_scraper.py` — الـ scraper التقليدي

**المكان:** `scraper/nhs_scraper.py`، 325 سطرًا. المصدر: NHS Inform A–Z.

### الثوابت والاستيرادات — الأسطر 1–38

- `argparse`: اختيار mode من terminal.
- `time`: throttling بين الطلبات.
- `re`: Regular Expressions لتنظيف/مطابقة العناوين.
- `cloudscraper`: session تقلّد متصفحًا وتتجاوز بعض الحمايات البسيطة.
- `BeautifulSoup`: قراءة HTML.
- `pandas`: إنشاء/تحديث CSV.
- `BASE_URL` و`INDEX_URL`: موقع NHS والجذر A–Z.
- `FIELDS`: الحقول الأربعة التي نريدها.
- `HEADERS`: User-Agent وheaders عادية حتى لا يبدو الطلب كـ bot بسيط.
- `S_PATTERNS`, `C_PATTERNS`, `W_PATTERNS`, `R_PATTERNS`: كلمات مفتاحية لتصنيف عنوان HTML إلى Symptoms/Causes/Warnings/Recommendations.

### الدوال

| الأسطر | الدالة | المدخلات → المخرجات | شرح مبسط ومثال |
| --- | --- | --- | --- |
| 39–51 | `fetch_html(url)` | URL → `BeautifulSoup` أو `None` | يحاول GET ثلاث مرات، timeout=20 ثانية، وينتظر 3 ثم 6 ثوانٍ عند الفشل. مثال: `fetch_html(INDEX_URL)` يعيد HTML صفحة A–Z. |
| 53–65 | `clean_text(text)` | نص HTML → نص أنظف | يحذف feedback وShare وآخر تحديث وSource ثم يدمج spaces. لا يترجم ولا يلخص. |
| 67–94 | `remove_repeated_runs(text, min_run=10)` | نص → نص | يبحث عن تسلسلات من 10 كلمات متكررة ويحذف النسخة التالية؛ يحارب تكرار HTML غير المقصود. |
| 96–103 | `clean_field(text, passes=3)` | نص → نص | يكرر `remove_repeated_runs` حتى يستقر النص أو يصل لـ3 passes. |
| 105–123 | `get_index()` | لا مدخلات → `{name: url}` | يزور A–Z، يقرأ كل `<a>`، يحتفظ فقط بروابط `/illnesses-and-conditions/` المناسبة. مثال: `{"Asthma": "https://.../asthma/"}`. |
| 125–134 | `_find_child_condition_links(soup, page_url)` | Hub page → حتى 6 child URLs | بعض صفحات NHS هي صفحة تجميع؛ يبحث داخل `<main>` عن روابط تبدأ بنفس URL. |
| 136–152 | `_collect_section_content(heading_tag)` | heading HTML → نص section | يجمع العناصر بعد `h2/h3` حتى heading بنفس المستوى أو أعلى. يقبل `p`, `ul`, `ol`, `dl`, `blockquote`, `table`. |
| 154–162 | `_extract_care_cards(soup)` | الصفحة → نص warning إضافي | يقرأ CSS classes المحتوية `care-card`, `urgent`, `alert`… لأن red flags قد تكون cards لا headings. |
| 164–239 | `parse_condition_page(soup, name, url, allow_hub_follow=True)` | soup صفحة شرط → dict | قلب الـ scraper: يحذف nav/footer/feedback، يحصل على `h1`، يمر على headings ويطابقها مع patterns، يضيف care cards للـ warnings، ويتبع children إن كانت صفحة hub قصيرة وفارغة. يعيد `condition`, fields, `url`, `is_hub_page`. |
| 241–263 | `run_full_scrape(output_path)` | path → CSV كامل | ينشئ index ثم يزور كل حالة. يحفظ checkpoint كل 10 حالات، وينتظر 1.5 s احترامًا للموقع. مثال: `python nhs_scraper.py --mode full --output nhs_conditions.csv`. |
| 265–311 | `run_fill_missing(input_path, output_path)` | CSV ناقص → CSV محسّن | يحدد rows التي بها null في واحد من `FIELDS`، يعيد العثور على URL بالاسم (ومطابقة normalised بديلة)، ويملأ **الخانة الناقصة فقط**؛ لا يكتب فوق بيانات موجودة. |
| 313–325 | `main()` | CLI → mode | يعرّف `--mode` (`full` أو `fill`) و`--input/--output`، ثم يستدعي الدالة المناسبة. block الأخير يجعله يعمل فقط عند التشغيل المباشر. |

**مثال ناتج `parse_condition_page`:**

```python
{
  "condition": "Asthma",
  "symptoms": "Wheezing ...",
  "causes": "...",
  "warnings": "...",
  "recommendations": "...",
  "url": "https://www.nhsinform.scot/...",
  "is_hub_page": False,
}
```

## `scraper/LLM_scraper.py` — ملء الحقول الناقصة عبر Mistral

**المكان:** `scraper/LLM_scraper.py`، 129 سطرًا.

هذا ليس endpoint الخاص بالـ GUI. إنه مرحلة data preparation تُستخدم بعد الـ scraper التقليدي، وتحتاج API URL لخادم Mistral منفصل.

| الأسطر | الكود | الشرح |
| --- | --- | --- |
| 12–32 | constants و`SYSTEM_PROMPT` | يحدد Mistral 7B، صفحات guide غير المناسبة، الحقول، وتعليمات صارمة: أرجع JSON بالمفاتيح المطلوبة فقط. الـ reference sample مأخوذ من صف مكتمل لضبط الأسلوب. |
| 34–36 | `is_guide_page(name)` | يمنع ملء صفحات من نوع “living with/caring for…” لأنها ليست تشخيصًا واحدًا واضحًا. |
| 38–45 | `build_reference(df)` | يأخذ أول صف كامل fields ويحضر snippets حتى 400 حرف لـ prompt. يجب وجود صف كامل، وإلا `iloc[0]` يفشل. |
| 47–64 | `call_llm(condition, missing_fields, reference, api_url)` | يبني system/user messages، يرسل `POST`، يتوقع `choices[0].message.content`، يحذف code fences ثم يعمل `json.loads`. مثال: قد يعيد `{"warnings":"..."}`. |
| 67–151 | `main()` | يقرأ CSV، يضيف `data_source` إن لم يوجد، يحدد الصفوف الناقصة، ويتعامل مع كل صف: guide/4 fields ناقصة → manual review؛ غير ذلك → LLM. يحفظ كل 10 rows، ويكتشف نص LLM مكررًا بين حالتين، ثم يحفظ final CSV وreview CSV بأمان. |

**مثال تشغيل:**

```powershell
python scraper\LLM_scraper.py `
  --input scraper\nhs_conditions.csv `
  --output scraper\nhs_conditions_final.csv `
  --review-output scraper\conditions_to_review.csv `
  --api-url https://<your-ngrok-url>/v1/chat/completions
```

لا تعني عبارة `data_source="llm_generated"` أن النص يُستخدم بلا مراجعة؛ الكود نفسه يصنع review list، وREADME يذكر مراجعة بشرية لاحقة.

## `scraper/transferToMongodb.ipynb` — نقل CSV إلى MongoDB

**المكان:** `scraper/transferToMongodb.ipynb`، 7 code cells.

| Cell | وظيفة السطور |
| --- | --- |
| 1 | يستورد `pandas`, `numpy`, `MongoClient`. |
| 2 | يقرأ `nhs_conditions_v2.csv`. لاحظ أن هذا الاسم لا يطابق اسم ملفات scraper الحالية؛ يحتاج تعديلًا أو rename قبل التشغيل. |
| 3 | `describe()` و`info()` لفحص الجودة. |
| 4 | يفتح `mongodb://localhost:27017/`، database `nhs_conditions_db` وcollection `conditions`. |
| 5 | `insert_data_to_mongodb(df, collection)`: يحول columns الصغيرة (`condition`, `symptoms`, ...) إلى schema كبيرة (`Condition_name`, `Symptoms`, ...)، ثم `insert_one` لكل صف. |
| 6 | ينفذ دالة الإدخال. |
| 7 | فارغة. |

**تحذير:** لا يوجد duplicate guard؛ تشغيل Cell 6 مرتين يضيف duplicates. الأفضل قبل التشغيل استخدام index unique على `Condition_name` أو upsert.

## `scraper/mistral_api_server(RUN IN KAGGLE).ipynb`

**المكان:** notebook، 3 code cells. هذا خادم تجريبي لاستضافة Mistral في Kaggle:

1. يثبت `pyngrok` وFlask.
2. يحمل النموذج وtokenizer لكن أسطر التحميل Commented؛ يجب فكها وتشغيلها.
3. يعرّف `generate_text`، و`messages_to_prompt`، route `/v1/chat/completions`، ثم ينشئ ngrok tunnel على 8000 ويشغل Flask في Thread. Endpoint يرجع OpenAI-like `choices[0].message.content`، وهو الشكل الذي يتوقعه `LLM_scraper.py`.

**أمان:** لا تضع ngrok token أو secrets داخل notebook قبل رفعه إلى GitHub.

## ملفات CSV داخل `scraper/`

| الملف | دوره | لماذا لا نفسر كل سطر كود؟ |
| --- | --- | --- |
| `nhs_conditions.csv` | المخرج الخام/الوسيط للـ conventional scraper، وقد يحوي nulls أو نصوص متعددة الأسطر. | كل row هو data record، لا instruction أو function. |
| `nhs_conditions_final.csv` | النسخة بعد ملء/مراجعة الحقول الناقصة. | يعرض نفس schema تقريبًا لكن بجودة/اكتمال أفضل. |

الأعمدة المهمة: `condition`, `symptoms`, `causes`, `warnings`, `recommendations`, وغالبًا `url`, `is_hub_page`, `data_source`. ظهور آلاف **الأسطر الفيزيائية** لا يعني آلاف records؛ حقول النص تحتوي newlines داخل quotes CSV.

---

# 5) `preprocessing_API/`: تنظيف النص وتصدير dataset

## `preprocessing_API/preprocessor.py`

**المكان:** 43 سطرًا.

| الأسطر | العنصر | الشرح |
| --- | --- | --- |
| 1–5 | imports | Regex وNLTK stopwords/tokenizer/lemmatizer. |
| 8–11 | `FALLBACK_STOP_WORDS` | قائمة صغيرة تعمل بدون تنزيل NLTK. |
| 14–21 | `_has_resource(path)` | يفحص local NLTK resource عبر `nltk.data.find` ولا ينزل شيئًا من الإنترنت. |
| 23–35 | `MedicalTextPreprocessor.__init__` | يحاول تحميل English stopwords؛ عند الفشل يستخدم fallback. ثم يقرر هل `punkt`, `punkt_tab`, `wordnet` متاحة. |
| 37–43 | `clean_text(text)` | يعيد `""` للنص الفارغ؛ lower-case؛ يحذف punctuation؛ يقطّع بـ NLTK أو regex؛ يحذف stopwords؛ يعمل lemmatization إن وُجد wordnet؛ يعيد string مفصولًا بمسافات. |

**مثال:**

```python
MedicalTextPreprocessor().clean_text("I have headaches, and I am feeling tired!")
# تقريبًا: "headache feeling tired"
```

النتيجة قد تختلف قليلًا حسب وجود NLTK data؛ هذا تصميم intentional حتى لا يتوقف المشروع عند HTTP 429.

## `preprocessing_API/create_usable_dataset.py`

**المكان:** 50 سطرًا.

الدالة الوحيدة `prepare_and_export_dataset(...)` (السطر 9):

1. تفتح MongoDB محليًا.
2. تعمل projection كي تحصل فقط على `_id=0` وحقول الحالة الخمسة.
3. تحول cursor إلى DataFrame؛ إن كان فارغًا تطبع رسالة وتعود.
4. تنشئ `MedicalTextPreprocessor`.
5. تضيف `cleaned_Symptoms`, `cleaned_Causes`, `cleaned_Warnings`, `cleaned_Recommendations` عبر `apply`.
6. تنشئ folder إن لم يوجد وتحفظ `dataset/usable_dataset.csv`.

**مثال:** `python preprocessing_API\create_usable_dataset.py` يستخدم defaults: DB `nhs_conditions_db`، collection `conditions`، output `dataset`.

## `preprocessing_API/app.py`

**المكان:** 22 سطرًا، Flask API صغير على port 5001.

- السطران 1–5: Flask + preprocessor global، لذلك لا ينشئ preprocessor مع كل request.
- السطور 7–20: route `POST /preprocess`. يطلب JSON مثل `{"text":"..."}`؛ إذا `text` غير موجود يرجع HTTP 400، وإلا يرجع raw وcleaned.
- 22: يبدأ server بـ `host="0.0.0.0"`, `port=5001`, `debug=True`.

**مثال request:**

```json
POST http://localhost:5001/preprocess
{"text": "I have persistent coughing and headaches."}
```

---

# 6) `dataset/`: الداتا التي يقرأها النظام

## `dataset/usable_dataset.csv`

الـ knowledge base الأساسي. يحتوي الأعمدة الخام:

```text
Condition_name, Symptoms, Causes, Warnings, Recommendations
```

وقد يحتوي أيضًا الأعمدة المنظفة التي أضافها preprocessing:

```text
cleaned_Symptoms, cleaned_Causes, cleaned_Warnings, cleaned_Recommendations
```

TF-IDF baseline يستخدم `cleaned_Symptoms` إن وُجدت، وإلا `Symptoms`. أما الواجهة فتعرض الحقول الخام حتى يبقى الكلام readable للمستخدم.

## `dataset/supplemental_conditions.csv`

ملف صغير، صف data واحد حاليًا: `ADHD in adults`. أضيف لأنه كان غائبًا من collection الأصلية بينما كان مطلوبًا لاختبار regression. أعمدته تشمل `source_url`، أي يمكن تتبع المصدر. لا يملك cleaned columns؛ نسخ `train_baseline.py` الحديثة يجب أن تقع back إلى raw `Symptoms` له.

---

# 7) `model_API/`: Baseline وFlask API وDeep Learning

## `model_API/train_baseline.py` — بناء الـ ML baseline

**المكان:** 134 سطرًا. الاسم القديم “Logistic Baseline” في بعض الواجهات لا يصفه بدقة؛ هذا الملف لا يدرب `LogisticRegression`، بل يبني retriever من `TF-IDF + FAISS`.

| الأسطر | الجزء | الشرح |
| --- | --- | --- |
| 13–22 | thread limits | يضبط OpenBLAS/OMP/MKL قبل NumPy وFAISS كي لا يتكرر خطأ memory allocation. |
| 31–36 | المسارات | يحدد base dataset، ملف ADHD supplemental، وثلاثة artifacts. |
| 39–41 | `_clean(value)` | يحول null إلى string فارغ ويصلح character replacement `�` إلى apostrophe. |
| 43–50 | `_dataset_hash(paths)` | يبني SHA-256 من **كل** data files الموجودة؛ لذلك إضافة ADHD تعيد بناء baseline تلقائيًا. |
| 52–57 | `_load_records()` | يقرأ `usable_dataset.csv` ويضيف `supplemental_conditions.csv` إن وجد ثم `concat` و`fillna("")`. |
| 60–125 | `train(force=False)` | إذا hash السابق مطابق لا يعمل rebuild. وإلا يتحقق من columns، يختار `cleaned_Symptoms` أو raw fallback للـ supplemental row، ينظف labels ويحذف duplicate labels، ثم TF-IDF 1–2 grams مع English stopwords وL2 norm. يحول matrix إلى `float32`، ينشئ `IndexFlatIP` ويحفظ index وrecords وjoblib artifact. |
| 127–134 | CLI | `--force` يجبر rebuild. |

**مثال تشغيل:**

```powershell
python model_API\train_baseline.py --force
```

**مهم:** هذا المسار يضم حاليًا base dataset + ADHD supplement؛ لذلك عدد records المبني يمكن أن يصبح 413.

## `model_API/deep_learning_pipeline.py` — التدريب الفعلي لـ RNN/LSTM

**المكان:** 304 سطرًا. هو التنفيذ البرمجي القابل للتشغيل من terminal. توجد أيضًا Notebook تعليمية في `model_API/notebooks/RNN_LSTM_Medical_Classification.ipynb` تشرح نفس architecture وتستدعي هذا المسار.

| الأسطر | الدالة/الفئة | شرح مبسط |
| --- | --- | --- |
| 25–30 | constants | مسار base dataset فقط، checkpoint root، regex tokens، ورمزا `<pad>` و`<unk>`. |
| 34–37 | `SymptomChunk` | dataclass immutable يحمل `tokens` و`label` (اسم الحالة). |
| 39–45 | `_clean`, `tokenize` | توحيد encoding ثم استخراج كلمات `[a-z0-9]+` من نص منخفض الأحرف. |
| 47–54 | `set_seed` | يثبت random/NumPy/PyTorch seed؛ يفعل CUDA seed إن وُجد لتقليل اختلاف النتائج بين runs. |
| 57–73 | `load_condition_documents` | يقرأ `Condition_name` و`Symptoms` من base CSV، يتجاهل labels/tokens الفارغة والduplicates، ويعيد list `(label, tokens)`. |
| 75–89 | `build_chunks` | يقسم symptoms الطويلة إلى windows متداخلة. مثال `chunk_size=64, stride=48`: chunk ثانٍ يبدأ بعد 48 token، فيحافظ على overlap 16 token. يحفظ آخر window كذلك. |
| 91–111 | `split_chunks_by_condition` | يجمع chunks لكل label؛ إن كان للحالة أكثر من chunk يحجز واحدًا أو أكثر للـ validation، ويترك chunk واحدًا على الأقل للتدريب. هذه ليست independent patient split. |
| 113–122 | `build_vocabulary` | يعد الكلمات في training chunks، ويبدأ vocabulary بـ `pad=0`, `unk=1`، ثم يضيف tokens التي frequency لها كافية. |
| 124–150 | `create_model` | يبني PyTorch model: `Embedding → Dropout(0.30) → nn.RNN أو nn.LSTM → Dropout → Linear(labels)`. يستخدم packed sequences كي لا تؤثر pads في state. `forward` يرجع logits، لا probabilities. |
| 152–172 | `create_dataset` | class داخلي `ChunkDataset`; كل item يعطي `input_ids`, `lengths`, `labels`. يقص النص عند `max_length` ويملأ الباقي بـ pad. |
| 174–188 | `evaluate` | `model.eval()` و`no_grad()`، يحسب held-out top-1 وtop-3 من logits. |
| 190–273 | `train` | يبني docs/chunks/split/vocab/labels وDataLoaders؛ يستخدم `AdamW`, `CrossEntropyLoss`, gradient clipping=1.0؛ في كل epoch يحفظ `epoch_NNN.pt` ويفضل `best.pt` وفق top-1 ثم metadata/history. |
| 275–304 | `main` | flags للتدريب: model, epochs, batch size, dimensions, chunks, validation, seed, device. `--model all` يدرب RNN ثم LSTM. |

**أمر التدريب الافتراضي:**

```powershell
python model_API\deep_learning_pipeline.py --model all
```

**المخرجات لكل model:** `best.pt`, `epoch_001.pt...`, `vocabulary.json`, `training_metadata.json` في `model_API/checkpoints/deep_learning/<rnn|lstm>/`.

**حد مهم في الداتا:** هذا الـ pipeline يقرأ `usable_dataset.csv` فقط، لذلك trained label count حاليًا 412؛ بينما baseline يضيف ADHD supplemental فيصل إلى 413. لهذا لا يصح اعتباره model مستقلًا أفضل من RAG، والـ API لديه quality guard.

## `model_API/deep_learning_inference.py` — تحميل checkpoint والتنبؤ

**المكان:** 78 سطرًا.

| الأسطر | العنصر | الشرح |
| --- | --- | --- |
| 15–17 | `DeepLearningModelUnavailable` | error مفهوم عندما checkpoint/vocabulary غير موجودة أو لا توجد labels مسموحة. |
| 19–36 | `_load_model(kind)` | cached (حتى نموذجين)؛ يقرأ `best.pt` و`vocabulary.json`، يعيد بناء architecture من checkpoint، يحمل weights، ويضع `eval()`. |
| 38–78 | `predict(kind, text, top_k=3, allowed_conditions=None)` | يحول text إلى token IDs مع `<unk>`/padding، يحسب softmax، ثم — عند وجود allow-list — يرتب فقط الحالات التي جاء بها RAG. يعيد top candidates وconfidence وtraining metrics. |

**لماذا `allowed_conditions` مهم؟** يضمن أن RNN/LSTM لا يرجع label ليس ضمن records التي استرجعها RAG للطلب الحالي.

## `model_API/app.py` — Flask API الموحد

**المكان:** 199 سطرًا ويعمل على port 5000.

| الأسطر | الدالة/route | الشرح |
| --- | --- | --- |
| 33–51 | `_fallback_tfidf_result` | إذا retriever لم يعمل، يقرأ joblib ويحسب sparse TF-IDF score يدويًا ويعيد record أعلى score. |
| 53–61 | `_predict` | baseline path الطبيعي: `get_retriever().search(text, top_k=3)` ثم `grounded_result`. |
| 63–84 | `_patient_context` | يتحقق أن age integer من 0 إلى 120 وgender لا يزيد 64 حرفًا. لا يضيف medical rules. |
| 86–117 | `_deep_prediction` | يسترجع 8 RAG candidates أولًا، يمرر أسماءهم كallow-list للـ deep model، يقرر إن كان checkpoint quality كافيًا: `held_out_chunk_top_1 >= 0.20`. إذا لا، يحتفظ بـ RAG top match ويضع `selection_mode="rag_quality_guard"`. |
| 120–129 | `GET /health` | health/status وartifact path ووجود semantic index. |
| 132–158 | `POST /predict` | baseline request: يطلب `text` ويقبل age/gender؛ يعيد structured record + candidates. |
| 161–194 | `POST /predict/deep-learning` | يقرأ `deep_model=rnn|lstm`، يعيد 400 للinput الخاطئ و409 للcheckpoint الغائب، ويعيد RAG candidates + model candidates + training metrics + selection mode. |
| 196–199 | startup | يبني baseline إن احتاج ثم يبدأ Flask. |

**مثال deep request:**

```json
POST /predict/deep-learning
{
  "text": "cracked and scaling skin between my toes",
  "deep_model": "lstm",
  "age": 24,
  "gender": "Male"
}
```

الـ `confidence` عند deep reranking هو score شرطي داخل RAG candidate set، وليس medical probability.

## `model_API/gridfs_handler.py` — legacy MongoDB model registry

يحتوي class `ModelGridFSHandler`:

- `__init__(mongo_uri, db_name)`: يفتح `MongoClient`, DB, `gridfs.GridFS` وcollection اسمها `models`.
- `save_model(...)`: يعمل `joblib.dump` في `BytesIO`، يرفع binary إلى GridFS، ثم يكتب metadata: name/type/gridfs_id/labels/metrics/created.
- `load_latest_model(model_name)`: يبحث عن آخر document بالاسم عبر sort `created:-1`، يقرأ binary من GridFS، ثم يعيد `(model, vectorizer, labels)`.

هذا الملف legacy ولا يستدعيه current Flask API. لذلك MongoDB ليس required للـ local baseline الحالي.

## `model_API/baseline_model.ipynb` — legacy Logistic Regression notebook

Notebook من 9 cells: يقرأ dataset، يدمج كل cleaned fields، يعمل TF-IDF، يدرب `LogisticRegression` على 100% من الداتا، ثم يقيس accuracy/F1 على **نفس training data** ويحفظ إلى GridFS. نتيجة 1.0 هنا ليست validation حقيقية. احتفظ به لتاريخ المشروع أو demo للـ workflow، لا كـ production baseline.

## `model_API/test.py`

Client اختبار بسيط يرسل `POST http://127.0.0.1:5000/predict` بالنص `can’t sleep at night` ثم يطبع condition وwarnings/causes/recommendations. يفترض API ناجحًا ولا يحتوي error handling.

## `model_API/artifacts/*` و`model_API/checkpoints/*`

- `artifacts/symptom_retrieval_baseline.joblib`: vectorizer + sparse matrix + records + hash.
- `artifacts/symptom_tfidf_faiss.bin`: FAISS `IndexFlatIP` binary.
- `artifacts/symptom_tfidf_records.pkl`: records المقابلة للـ index.
- `checkpoints/deep_learning/rnn/` و`lstm/`: weights per epoch وbest weight وvocabulary وtraining metadata. لا توجد أسطر مصدرية قابلة للشرح لأنها serialized data files.

تم اختبار RNN وLSTM checkpoint endpoint في النسخة الحالية؛ كلاهما رجع HTTP 200. لكن الـ metadata المسجل ضعيف جدًا (RNN held-out Top-1 ≈ 0.35% وLSTM = 0%)، ولذلك يظهر `rag_quality_guard` ويظل RAG هو المصدر الذي يختار الحالة المعروضة.

---

# 8) `rag_llm_API/`: الاسترجاع والـ RAG

## `rag_llm_API/retriever.py` — الملف الأهم وقت inference

**المكان:** 211 سطرًا. يحمل مسارين للاسترجاع.

| الأسطر | الدالة/الفئة | الشرح |
| --- | --- | --- |
| 12–23 | thread-limit environment | يضع limits قبل NumPy/FAISS لتجنب تخصيص threads/ذاكرة زائدة. |
| 25–34 | paths/constants | semantic index داخل `faiss_index/` وTF-IDF artifacts داخل `model_API/artifacts/`. |
| 37–40 | `_clean_text` | null/encoding cleanup. |
| 42–47 | `_field(chunk, name)` | Regex يقرأ section محددًا من string منظم بالشكل `Symptoms: ...\nCauses: ...`. |
| 49–52 | `_condition_from_chunk` | يقرأ أول السطر `Medical Condition: X.`. |
| 54–63 | `_tokens`, `_lexical_score` | يقيس نسبة كلمات query (3+ حروف) المشتركة مع النص. مثال: query tokens `{itching, scaling, toes}` و2 مشتركة = `2/3`. |
| 65–67 | `_normalise_name` | يحذف punctuation/case للمقارنة؛ `Athlete’s foot` و`athletes-foot` يصبحان نفس الشكل تقريبًا. |
| 69–119 | `ConditionRetriever` | semantic path. يحمل MiniLM وsemantic FAISS/chunks. `search` يشفّر query، يبحث كل vectors، يمزج `0.70*vector + 0.30*lexical`، ثم يفك fields من chunk ويرجع top_k. لا يستورد `SentenceTransformer` إلا عند اختيار هذا backend. |
| 121–182 | `TfidfFaissRetriever` | local default. يحمل joblib وFAISS TF-IDF. `search` يحول query بالـ saved vectorizer، ثم يبحث FAISS أو sparse fallback إن لم يوجد binary. `get_condition` يبحث exact normalised label ويعيد fields المصدرية. |
| 184–198 | `grounded_result` | يضمن response schema ثابت: condition, fields, confidence `%`, retrieval candidates. يملأ field الناقص بـ `Not available`. |
| 200–204 | `match_candidate` | يقارن إجابة LLM مع names المرشحة فقط. |
| 206–211 | `get_retriever` | cache بحجم 1. Environment `RAG_EMBEDDING_BACKEND=sentence_transformer` يختار semantic path؛ الافتراضي `tfidf_faiss`. |

**مثال:**

```python
from rag_llm_API.retriever import get_retriever
get_retriever().search("itching and scaling between toes", top_k=3)
```

يعيد list dictionaries، وليس final diagnosis.

## `rag_llm_API/build_faiss_index.py` — بناء semantic index الاختياري

**المكان:** 301 سطرًا.

| الأسطر | الجزء | ماذا يفعل؟ |
| --- | --- | --- |
| 1–40 | imports + thread env | يضبط threads قبل NumPy/FAISS، يستورد Mongo/CSV/SentenceTransformer، ويجبر terminal Windows على UTF-8. |
| 43–70 | `header`, `step`, `info`, `success`, `warn`, `error` | helpers للألوان والطباعة؛ لا تؤثر على البيانات. |
| 76–88 | paths وMongo settings | ينشئ `faiss_index/` ويحدد artifacts، Mongo URI، وfallback `dataset/usable_dataset.csv`. |
| 94–126 | `fetch_records()` | يحاول Mongo مع ping في 5 seconds؛ إن فشل يقرأ CSV، يتحقق من columns ويزيل duplicate conditions. |
| 129–181 | `build_chunks(records)` | لكل condition يصنع `search_chunk` = name + symptoms فقط (للأدق في search)، و`full_chunk` = كل fields (للـ context)، وmetadata dict. |
| 185–199 | `embed_chunks(chunks)` | يحمل `all-MiniLM-L6-v2` ويحوّل search chunks إلى matrix embeddings، batch 64. |
| 202–214 | `create_faiss_index(embeddings)` | L2 normalize ثم `IndexFlatIP`; inner product بعد normalize يساوي cosine similarity. |
| 217–246 | `sanity_check` | يرسل 5 queries معروفة ويطبع top match/score؛ فحص دخاني لا metric report. |
| 248–262 | `save_index` | يحفظ binary FAISS وchunks/metadata كـ pickles. |
| 265–301 | main pipeline | ينفذ Fetch → chunks → embeddings → index → sanity → save، ويحذف temp `_inspect_db.py` إن وجد. |

**متى تشغله؟** فقط عند تحديث records إن أردت semantic backend. default TF-IDF baseline لا يحتاج MiniLM index للبحث اليومي.

## `rag_llm_API/manual_check.py` — اختبار CLI قديم end-to-end

**المكان:** 490 تقريبًا سطرًا. هو أداة يدوية، ليس API production.

| الأسطر | دوال | الشرح |
| --- | --- | --- |
| 55–80 | logging helpers | نفس فكرة printing الملون. |
| 92–122 | `load_rag_components()` | يتحقق من files، يقرأ FAISS/chunks، ويحمل MiniLM. |
| 125–133 | `_keyword_similarity` | overlap بين كلمات query/chunk. |
| 135–216 | `retrieve_relevant_context` | يبحث كل vectors ثم يمزج `0.5*vector + 0.5*keyword`، يرفض ما تحت threshold=0.30، ويرجع context string فيه top candidates. |
| 219–245 | `extract_json_block` | يحذف `<think>`, يقرأ ```json``` أو object عادي. |
| 248–268 | `generate_mistral` | POST إلى custom `/generate` مع bearer key وngrok header. |
| 270–301 | `generate_qwen` | POST إلى `/api/chat` بشكل Ollama. |
| 303–327 | `SYSTEM_TEMPLATE`, `USER_TEMPLATE` | prompts يقولان للـ LLM ألا يخلط facts بين candidates وأن يعيد JSON. |
| 329–345 | `build_prompts` | يملأ LangChain `PromptTemplate`. |
| 347–447 | `run_rag_pipeline` | orchestration: retrieve → context display → LLM call → parse output → display. عند failure يرجع error dictionary. |
| 450–474 | `_display_result`, `_error_result` | formatting للـ terminal وfallback response. |
| 476–النهاية | CLI | يحمل components، يسأل المستخدم provider وURL/query، ثم يشغل pipeline. |

**ملاحظة تصميمية:** هذا الملف يسمح للـ LLM بإعادة صياغة fields بعد retrieval، بينما الـ GUI الحديث يقيد الواجهة أكثر ويعرض stored record. لذلك لا تستخدمه كمرجع safety النهائي.

## `rag_llm_API/mistral-api_(RUN IN KAGGEL).ipynb`

Notebook تشغيل Mistral-Nemo في Kaggle:

1. Cell 1: imports Kaggle boilerplate.
2. Cell 3: يثبت FastAPI/Uvicorn/ngrok/transformers/accelerate/bitsandbytes.
3. Cell 4: يحمل `Mistral-Nemo-Instruct-2407` quantized 4-bit بـ `BitsAndBytesConfig` ليستوعبه Tesla T4.
4. Cell 5: `generate_text` يمرر prompt للموديل ويقص prompt من الناتج.
5. Cell 6: placeholders لـ `NGROK_TOKEN` و`API_KEY`؛ يجب تغييرها كsecrets لا حفظها.
6. Cell 7: endpoint `POST /generate` ويتحقق من `Authorization: Bearer <API_KEY>` ثم يعيد `{"response": ...}`.
7. Cell 8: يختار free port، يفتح ngrok، ثم يشغل Uvicorn thread.

## `rag_llm_API/NHS_Ollama_(RUN IN COLAB).ipynb`

Notebook يفتح Qwen/Ollama في Colab:

1. يثبت `zstd` وOllama.
2. يشغل `ollama serve` على `0.0.0.0:11434`.
3. يحمل `qwen3:4b`.
4. يثبت pyngrok.
5. يقرأ token، يغلق tunnels قديمة، ويفتح URL عام لـ port 11434.

---

# 9) `gui/`: واجهة المستخدم والـ clients

## `gui/app.py` — واجهة Streamlit

**المكان:** 352 سطرًا. لا يحتوي model inference بنفسه؛ ينسق إدخال المستخدم ويستدعي `analyse_symptoms`.

| الأسطر | العنصر | الشرح |
| --- | --- | --- |
| 1–25 | imports و`set_page_config` | HTML escaping لمنع HTML injection في cards، env vars، Streamlit، ودوال clients. page title/icon/layout/sidebar. |
| 27–212 | `add_page_style()` | CSS فقط: theme طبي فاتح، gradient hero، sidebar، inputs، buttons، cards، responsive mobile وreduced-motion. لا توجد business logic هنا. |
| 214–221 | `show_result_card(title, value)` | يعمل `html.escape` للعنوان/القيمة ثم يعرضها كـ result card؛ escaping مهم حتى لا ينفذ النص القادم من API كـ HTML. |
| 224–239 | `render_branding()` | يرسم hero: local assistant، grounding trust chips. لا يعرض الشريط العلوي الذي حُذف لاحقًا. |
| 242–245 | setup | يستدعي CSS وhero ثم warning طبي ثابت. |
| 247–305 | sidebar | يقرأ `LOCAL_MODELS_ONLY`; عند true يظهر 3 local options فقط. عند false يضيف Mistral/Qwen وحقول URL/key. ثم age (0–120) وgender (`Not specified`, `Female`, `Male`). |
| 307–316 | input/button | `st.text_area` للأعراض وbutton `Analyse symptoms`. |
| 318–337 | validation/mapping | يرفض input فارغ؛ يحول label UI إلى internal key مثل `deep_lstm`; يظهر spinner ويستدعي `analyse_symptoms`. |
| 339–371 | result UI | يعرض success/confidence وdeep training metrics وquality guard notice ثم 5 cards والـ technical JSON في expander. |

**مثال تحويل:** اختيار `Deep Learning (RNN)` في الواجهة يصبح `model="deep_rnn"` في `model_clients.py`.

## `gui/app_local_models.py` — بوابة local-only

**المكان:** 9 أسطر.

```python
os.environ["LOCAL_MODELS_ONLY"] = "1"
runpy.run_path(..."app.py", run_name="__main__")
```

لا يكرر الواجهة. فقط يضع flag ثم يشغل نفس `app.py`، فتختفي Mistral/Qwen من options وfields. يجب تشغيله عبر launcher لا كحل لمجلد project ناقص.

## `gui/model_clients.py` — طبقة الاتصال والفحص

**المكان:** 382 سطرًا.

| الأسطر | الدالة | شرح/مثال |
| --- | --- | --- |
| 18–22 | constants | base URL الافتراضي `http://localhost:5000`، timeouts 90/45 ثانية. |
| 25–27 | `ModelRequestError` | exception آمن للعرض في GUI بدل traceback. |
| 29–72 | `analyse_symptoms(...)` | router: baseline → `_call_baseline`; deep → `_call_deep_learning`; Qwen/Mistral → استرجاع candidates أولًا ثم LLM، ثم `match_candidate` أو أول candidate. |
| 74–87 | `_retrieve_candidates` | lazy-import للretriever كي يفتح local UI حتى إن كان API لم يبدأ، ثم search top-3. |
| 89–116 | `_call_baseline` | POST إلى `/predict` بالـ symptoms/age/gender ويحوّل response إلى schema GUI. |
| 118–151 | `_call_deep_learning` | POST إلى `/predict/deep-learning` مع `deep_model="rnn"` أو `"lstm"`; يتوقع training metrics وselection mode من Flask API. |
| 153–169 | `_candidate_context` | يحول candidates إلى نص bounded lengths كي لا يصبح prompt ضخمًا. |
| 171–199 | `_selection_prompt` | system prompt يمنع اختراع condition/facts ويطلب JSON بالاسم exact، وuser prompt يحمل symptoms + optional context. |
| 201–258 | `_call_qwen` | يجرب `/api/chat` ثم `/api/generate`. temperature=0، ويجمع errors ليعطي رسالة ذات معنى. |
| 260–306 | `_call_mistral` | يدعم official Mistral (`/v1/chat/completions` + API key) وcustom ngrok Flask (`/generate`). |
| 308–321 | `_response_text` | يقرأ response من أشكال API مختلفة: `message.content`, `choices`, `response`, `generated_text`… |
| 323–337 | `_selected_condition` | يستخرج JSON؛ إن أخفق يسمح فقط باسم candidate موجود نصيًا. وإلا error، لا يخترع fallback name. |
| 339–355 | `_extract_json` | يحذف Qwen `<think>`, ثم يفحص fenced JSON ثم JSON object raw. |
| 357–372 | helpers | `_require_url` يتحقق من URL، `_json_response` يحمي من non-JSON، `_error_message` يوحد رسالة error. |

**مثال مسار Qwen:** user → `_retrieve_candidates` → prompt يضم top-3 → Qwen يقول `{"condition":"Asthma"}` → `match_candidate` يتأكد أن Asthma ضمن الثلاثة → GUI يعرض stored fields من candidate.

## `gui/README.md`

دليل قديم/مختصر للواجهة: يذكر 3 models ويقول `Logistic Baseline`. لا يذكر local-only mode أو deep options، لذا يجب اعتباره documentation يحتاج تحديثًا وليس مصدر الحقيقة النهائي.

## `gui/requirements.txt`

سطران فقط: `streamlit` و`requests`. مناسب لتشغيل UI وحده، لكنه غير كافٍ للمشروع الكامل؛ local/production تحتاج dependencies في root requirements.

---

# 10) `evaluation/` و`reports/`: تقييم قابل للتكرار

## `evaluation/baseline_regression_cases.csv`

10 prompts ثابتة: columns هي `case_id`, `expected_condition`, `prompt`, `focus`. ليست training data، بل regression tests. مثال ADHD يمنع رجوع خطأ Wolff-Parkinson-White بدل ADHD.

## `evaluation/evaluate_baseline.py`

**المكان:** 87 سطرًا.

| الأسطر | العنصر | الشرح |
| --- | --- | --- |
| 28–30 | `_normalise` | يقارن names من دون case/punctuation/encoding noise. |
| 32–74 | `evaluate(cases_path, top_k)` | يحمل `TfidfFaissRetriever`، يقرأ CSV، يبحث top-k لكل prompt، يحسب rank، top-1 correctness، top-k correctness، reciprocal rank ثم summary. |
| 76–86 | `main` | flags `--cases`, `--top-k`؛ top-k لا يقل عن واحد؛ يطبع JSON readable. |

**Metrics:** Top-1 = المرشح الأول صح؟ Top-3 recall = expected موجود في 3؟ MRR = 1/rank متوسطًا.

## `evaluation/evaluate_retrieval.py`

Wrapper متوافق مع command قديم. ينادي `evaluate_baseline.evaluate` ثم يطبع results كنص بدل JSON. `--top-k` من 1 إلى 10.

## `evaluation/retrieval_cases.json`

10 test objects بديلة schema: `id`, `expected_condition`, `query`. حاليًا evaluator الرئيسي يقرأ CSV وليس هذا JSON؛ الملف مفيد للاختبار اليدوي/المستقبلي.

## `reports/experimentation_evaluation_report.md`

تقرير نتائج baseline:

- يصرح أنه functional regression لا clinical study.
- config: 413 records، symptom-only TF-IDF 1–2 grams، FAISS `IndexFlatIP`، 10 fixed cases.
- نتائج مسجلة في 17 أغسطس 2026: Top-1 = 80%، Top-3 recall = 100%، MRR@3 = 0.90.
- يذكر صراحة limitation: condition document واحد تقريبًا لكل label، فلا يصح classifier train/test عشوائي.
- يحدد next steps: clinician-reviewed independent set وabstention/emergency tests.

---

# 11) `docs/`: وثائق Deep Learning

## `docs/DEEP_LEARNING_TRAINING.md`

يصف architecture للـ RNN/LSTM: tokens → Embedding → RNN/LSTM → Dropout → Dense، ومسار checkpoints، وتحذير أن held-out chunks ليست patient validation، وأن quality guard يبقي RAG top match لو Top-1 أقل من 20%.

الـ Notebook المشار إليها موجودة الآن في `model_API/notebooks/RNN_LSTM_Medical_Classification.ipynb`، وقد تم التحقق من صحة JSON وPython code cells الخاصة بها. للتدريب repeatable من terminal استخدم `model_API/deep_learning_pipeline.py`؛ والـ Notebook هي المسار التفاعلي لعرض المراحل نفسها.

---

# 12) قائمة الفحص قبل أي تشغيل أو عرض

1. تأكد أن `model_API/` موجود؛ هو موجود في النسخة الحالية ويحتوي artifacts وRNN/LSTM checkpoints.
2. شغّل `python -m pip install -r requirements.txt` في `.venv`.
3. للـ local TF-IDF path، شغّل `model_API/train_baseline.py` وتأكد من artifacts.
4. نفّذ `python evaluation/evaluate_baseline.py --top-k 3` بعد artifact build.
5. شغّل Flask 5000 ثم Streamlit 8501.
6. لا تعرض نتيجة retrieval كـ diagnosis؛ confidence هو similarity/ranking score وليس احتمالًا طبيًا.
7. لا تضع ngrok token أو Mistral API key داخل Git/notebooks.
8. ملف `gui/app_local_models.py` موجود لتشغيل واجهة بلا Qwen/Mistral، لكن ملفا `run_local_models_only.bat` و`requirements_local_models.txt` غير موجودين حاليًا في الجذر. استخدم launcher الكامل أو أعد إضافة هذين الملفين قبل إرسال نسخة local-only لصديق.

## مثال كامل لمسار Athlete’s foot

1. المستخدم يكتب: `itching, burning, cracked scaling skin between the toes`.
2. GUI يمرر النص إلى `/predict`.
3. API ينادي `get_retriever().search(text, top_k=3)`.
4. TF-IDF يركز على كلمات مثل `cracked`, `scaling`, `toes` ويبحث cosine similarity في local records.
5. candidate الأعلى يكون `Athlete's foot` عندما تكون الداتا مطابقة.
6. `grounded_result` يأخذ Causes/Warnings/Recommendations من row Athlete’s foot نفسه.
7. GUI يعرض cards؛ لا يطلب من LLM تأليف محتوى طبي جديد.

## ما الذي لا يمكن أن يفعله المشروع بأمان؟

- لا يشخص مرضًا طبيًا ولا يقيس medical probability حقيقية.
- لا يملك demographic eligibility table، لذلك age/gender context لا يجب أن يغير facts المحفوظة.
- لا يملك abstention rule قويًا؛ prompt غير متعلق قد يحصل على أقرب record رغم أنه ليس match مناسبًا.
- لا يجوز اعتبار training accuracy للـ legacy Logistic notebook أو held-out chunks للـ RNN/LSTM دليل clinical performance.
