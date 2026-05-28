import sys
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="LeukoCheck v3.1",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.core import FEATURE_ORDER, LeukoCheckEngine

logging.basicConfig(level=logging.INFO)

RUSSIAN_NAMES = {
    'age':       'Возраст (лет)',
    'gender':    'Пол',
    'bmi':       'ИМТ (кг/м²)',
    'smoking':   'Курение',
    'diabetes':  'Сахарный диабет',
    'hypert':    'Арт. гипертензия',
    'sbp':       'САД (мм рт.ст.)',
    'dbp':       'ДАД (мм рт.ст.)',
    'hr':        'ЧСС (уд/мин)',
    'PP':        'Пульсовое давление',
    'wbc':       'Лейкоциты',
    'neut_pct':  'Нейтрофилы (%)',
    'lymph_pct': 'Лимфоциты (%)',
    'mono_pct':  'Моноциты (%)',
    'platelets': 'Тромбоциты',
    'NLR':       'Нейтрофилы / Лимфоциты',
    'MLR':       'Моноциты / Лимфоциты',
    'PLR':       'Тромбоциты / Лимфоциты',
    'ISNM':      'Нейтрофилы / Моноциты',
}

@st.cache_resource
def get_engine():
    return LeukoCheckEngine()

engine = get_engine()

st.title("🫀 LeukoCheck: Оценка риска сердечно-сосудистых заболеваний")
st.markdown(
    "*Система оценки риска ИБС и инсульта "
    "на основе общего анализа крови и гемодинамических показателей*"
)
st.warning(
    "⚠️ **Важно**: результат не является медицинским диагнозом. "
    "Модель обучена на данных **NHANES 2017-2018** (N=3057, AUC 0.775) "
    "и не прошла клиническую валидацию на российской популяции. "
    "Требуется консультация врача."
)

if not engine.is_ready:
    st.error("❌ Модель не загружена. Проверьте наличие файлов в папке `models/`.")
    st.stop()

st.sidebar.header("📋 Данные пациента")

with st.sidebar.expander("👤 Демография", expanded=True):
    age    = st.slider("Возраст (лет)", 18, 100, 55)
    gender = st.selectbox("Пол", ["Женский", "Мужской"], index=0)

    col_h, col_w = st.columns(2)
    with col_h:
        height = st.number_input("Рост (см)", 100.0, 220.0, 170.0, step=1.0)
    with col_w:
        weight = st.number_input("Вес (кг)", 30.0, 200.0, 70.0, step=0.5)

    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label = "🔹 Недостаточный"
    elif bmi < 25:   bmi_label = "🟢 Норма"
    elif bmi < 30:   bmi_label = "🟡 Избыточный"
    else:            bmi_label = "🔴 Ожирение"
    st.caption(f"ИМТ: **{bmi:.1f}** кг/м² — {bmi_label}")

with st.sidebar.expander("⚠️ Факторы риска", expanded=True):
    smoking      = st.checkbox("Курение",                  value=False)
    diabetes     = st.checkbox("Сахарный диабет",          value=False)
    hypertension = st.checkbox("Артериальная гипертензия", value=False)

with st.sidebar.expander("❤️ Гемодинамика", expanded=True):
    sbp = st.number_input("САД (мм рт.ст.)", 60,  260, 120)
    dbp = st.number_input("ДАД (мм рт.ст.)", 40,  160,  80)
    hr  = st.number_input("ЧСС (уд/мин)",    30,  200,  72)

with st.sidebar.expander("🔬 Лабораторные показатели", expanded=True):
    wbc   = st.number_input("Лейкоциты (×10⁹/л)", 0.5, 50.0, 7.0, step=0.1)
    neut  = st.number_input("Нейтрофилы (%)",      0,   100,  60)
    lymph = st.number_input("Лимфоциты (%)",        0,   100,  30)
    mono  = st.number_input("Моноциты (%)",         0,   100,   6)

    st.divider()
    platelets_known = st.checkbox("Тромбоциты известны", value=False,
                                  help="Включите для точного расчёта индекса воспаления")
    platelets = None
    if platelets_known:
        platelets = st.number_input("Тромбоциты (×10⁹/л)", 20, 1500, 250)

calc_button = st.sidebar.button("🔍 Рассчитать риск", type="primary", use_container_width=True)

if calc_button:
    patient_data = {
        'age':          age,
        'gender':       1 if gender == "Мужской" else 0,
        'bmi':          bmi,
        'smoking':      smoking,
        'diabetes':     diabetes,
        'hypertension': hypertension,
        'sbp':          sbp,
        'dbp':          dbp,
        'hr':           hr,
        'wbc':          wbc,
        'neutrophils':  neut,
        'lymphocytes':  lymph,
        'monocytes':    mono,
    }
    if platelets is not None:
        patient_data['platelets'] = platelets

    result = engine.predict_risk(patient_data)

    if 'error' in result:
        st.error(f"❌ {result['error']}")
        st.stop()

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Результат")
        risk_pct = result['risk_probability']
        if risk_pct >= 70:
            st.error(f"## 🔴 {risk_pct}%")
            st.caption("Высокий риск")
        elif risk_pct >= 30:
            st.warning(f"## 🟡 {risk_pct}%")
            st.caption("Умеренный риск")
        else:
            st.success(f"## 🟢 {risk_pct}%")
            st.caption("Низкий риск")

    with col2:
        st.subheader("💡 Рекомендации")
        st.info(result['recommendation'])

        st.subheader("🔬 Индексы воспаления")
        abs_neut_ui  = wbc * neut  / 100
        abs_lymph_ui = wbc * lymph / 100
        abs_mono_ui  = wbc * mono  / 100

        nlr_val  = abs_neut_ui  / abs_lymph_ui if abs_lymph_ui > 0 else float('nan')
        mlr_val  = abs_mono_ui  / abs_lymph_ui if abs_lymph_ui > 0 else float('nan')
        isnm_val = abs_neut_ui  / abs_mono_ui  if abs_mono_ui  > 0 else float('nan')

        c1, c2, c3 = st.columns(3)
        c1.metric("Нейтр. / Лимф.", f"{nlr_val:.2f}",
                  help="Норма: < 3.0. Повышение отражает воспалительный сдвиг.")
        c2.metric("Моноц. / Лимф.", f"{mlr_val:.2f}",
                  help="Норма: < 0.3. Маркер хронического воспаления.")
        c3.metric("Нейтр. / Моноц.", f"{isnm_val:.2f}",
                  help="Индекс системного нейтрофильно-моноцитарного воспаления.")

        plt_val  = platelets if platelets_known else 250
        sii_val  = (abs_neut_ui * plt_val) / abs_lymph_ui if abs_lymph_ui > 0 else float('nan')
        sii_note = "" if platelets_known else " (тромбоциты не введены, приблизительно)"
        st.caption(f"Системный индекс воспаления (SII): **{sii_val:.1f}**{sii_note}")

    with st.expander("🧠 Что повлияло на результат?", expanded=False):
        try:
            import shap

            X_df = engine.calculate_all_features(patient_data)

            background = engine.shap_background_scaled
            explainer  = shap.LinearExplainer(engine.model, background)
            X_scaled   = engine.scaler.transform(X_df)
            shap_vals  = explainer.shap_values(X_scaled)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_vals = np.asarray(shap_vals).flatten()

            top_n      = 8
            sorted_idx = np.argsort(np.abs(shap_vals))[-top_n:]
            top_vals   = shap_vals[sorted_idx]
            top_names  = [RUSSIAN_NAMES[FEATURE_ORDER[i]] for i in sorted_idx]
            top_data   = X_df.values[0][sorted_idx]

            labels = []
            for name, val in zip(top_names, top_data):
                val_str = "Да" if val == 1.0 else ("Нет" if val == 0.0 else f"{val:.1f}")
                labels.append(f"{name}:  {val_str}")

            colors = ["#D94F4F" if v > 0 else "#4A86C8" for v in top_vals]

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor('#FAFAFA')
            ax.set_facecolor('#FAFAFA')

            bars = ax.barh(range(len(top_vals)), top_vals,
                           color=colors, height=0.55, edgecolor='none')

            for bar, val in zip(bars, top_vals):
                sign  = "▲ повышает риск" if val > 0 else "▼ снижает риск"
                x_pos = val + 0.008 if val > 0 else val - 0.008
                ha    = 'left' if val > 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        sign, va='center', ha=ha, fontsize=8.5, color='#555555')

            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=10, color='#333333')
            ax.axvline(0, color='#AAAAAA', linewidth=1.0, linestyle='--')
            ax.set_xlabel("Сила влияния на прогноз", fontsize=10, color='#555555')
            ax.set_title("Факторы, повлиявшие на результат",
                         fontsize=13, fontweight='bold', pad=14, color='#222222')
            x_max = max(abs(top_vals)) * 2.2
            ax.set_xlim(-x_max, x_max)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, colors='#555555')
            ax.xaxis.set_visible(False)

            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor='#D94F4F', label='Повышает риск'),
                    Patch(facecolor='#4A86C8', label='Снижает риск'),
                ],
                loc='lower right', fontsize=9, framealpha=0.7, edgecolor='none'
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Три главных фактора:**")
            impacts = sorted(
                zip([RUSSIAN_NAMES[f] for f in FEATURE_ORDER], X_df.values[0], shap_vals),
                key=lambda x: abs(x[2]), reverse=True
            )
            for i, (name_ru, val, impact) in enumerate(impacts[:3], 1):
                direction = "повышает риск 🔺" if impact > 0 else "снижает риск 🔻"
                val_str = "Да" if val == 1.0 else ("Нет" if val == 0.0 else f"{val:.1f}")
                st.write(f"**{i}. {name_ru}** = {val_str} — {direction}")

            st.caption(
                "Длина полосы отражает силу влияния фактора. "
                "Красный — тянет риск вверх, синий — вниз."
            )

        except ImportError:
            st.warning("⚠️ Модуль `shap` не установлен: `pip install shap`")
        except Exception as e:
            st.warning(f"⚠️ Не удалось построить объяснение: {e}")

with st.expander("ℹ️ О модели"):
    st.markdown("""
**LeukoCheck v3.1** — прототип системы оценки риска болезней системы кровообращения.

**Методология:**
- Воспалительные индексы: NLR, MLR, PLR, ИСНМ
- Модель: логистическая регрессия, NHANES 2017-2018 (N=3057)

**Метрики (тестовая выборка, bootstrap 95% CI):**
- AUC-ROC: 0.775 [95% CI: 0.656–0.867]
- CV AUC (5-fold): 0.795 ± 0.020
- Brier Score: 0.169

**Ограничения:**
- Обучена на американской популяции NHANES — валидация на российских данных СибГМУ в процессе
- Результат не является диагнозом и не заменяет консультацию врача
""")

st.divider()
st.caption("LeukoCheck v3.1 · прототип для исследовательских целей · ООО ВОГНЕР ГРУПП")