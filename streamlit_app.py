# streamlit_app.py
"""
AI Medical Assistant — Streamlit App
Features:
- Logistic Regression classifier (symptom -> diagnosis)
- Forward chaining (rule engine)
- Backward chaining (proof trace)
- BFS and DFS traversal on rule graph (returns paths)
- Simple SQLite patient DB: save & retrieve patient records
- Explainability: rule traces & classifier token contributions

To run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from collections import deque
import json

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="AI Medical Assistant", layout="wide")
DB_PATH = "patients_demo.db"

# ---------------------------
# Utility: Simple SQLite wrapper
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            symptoms TEXT,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient(name, age, gender, symptoms, notes):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO patients (name, age, gender, symptoms, notes) VALUES (?,?,?,?,?)',
              (name, age, gender, symptoms, notes))
    conn.commit()
    conn.close()

def list_patients():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)
    conn.close()
    return df

def get_patient(pid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id=?", (pid,))
    row = c.fetchone()
    conn.close()
    return row

# Ensure DB exists
init_db()

# ---------------------------
# 1) Training data and classifier (cached resource)
# ---------------------------
@st.cache_resource
def train_classifier():
    # Synthetic small dataset for demo; replace with clinical data for production
    samples = [
        ("fever cough headache", "Flu"),
        ("fever rash itchy_spots", "Viral Exanthem"),
        ("headache stiff neck fever", "Meningitis"),
        ("chest pain shortness breath", "Cardiac Event"),
        ("fatigue weight loss night sweats", "TB suspect"),
        ("sore throat swollen tonsils fever", "Pharyngitis"),
        ("cough wheezing shortness breath", "Asthma Exacerbation"),
        ("abdominal pain vomiting diarrhea", "Gastroenteritis"),
        ("runny nose fever", "Cold"),
        ("nausea vomiting", "Gastroenteritis")
    ]
    X_text = [s for s, l in samples]
    y = [l for s, l in samples]
    vec = CountVectorizer()
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    return clf, vec

clf, vectorizer = train_classifier()

# ---------------------------
# 2) Rule set (forward/backward chaining)
# ---------------------------
RULES = [
    {"id": "R1", "if": ["fever", "cough"], "then": "Flu", "weight": 1.0},
    {"id": "R2", "if": ["fever", "rash"], "then": "Viral Exanthem", "weight": 0.8},
    {"id": "R3", "if": ["headache", "stiff neck", "fever"], "then": "Meningitis", "weight": 1.2},
    {"id": "R4", "if": ["chest pain", "shortness of breath"], "then": "Cardiac Event", "weight": 1.5},
    {"id": "R5", "if": ["fatigue", "weight loss", "night sweats"], "then": "TB suspect", "weight": 1.0},
    {"id": "R6", "if": ["fever", "sore throat"], "then": "Pharyngitis", "weight": 0.9},
    {"id": "R7", "if": ["cough", "wheezing"], "then": "Asthma Exacerbation", "weight": 1.0},
    {"id": "R8", "if": ["abdominal pain", "vomiting"], "then": "Gastroenteritis", "weight": 0.9}
]

def forward_chaining(symptoms):
    """Return fired rules and disease scores."""
    symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    fired = []
    scores = {}
    for r in RULES:
        matches = sum(1 for a in r["if"] if a in symptoms)
        if matches > 0:
            score = (matches / len(r["if"])) * r.get("weight", 1.0)
            scores[r["then"]] = max(scores.get(r["then"], 0), score)
            fired.append({
                "rule_id": r["id"],
                "if": r["if"],
                "matched_count": matches,
                "matched_items": [a for a in r["if"] if a in symptoms],
                "then": r["then"],
                "score": round(score, 3)
            })
    return {"fired": fired, "scores": scores}

def backward_chaining(goal, symptoms):
    """Given goal (disease) and provided symptoms, return proof trace."""
    if not goal:
        return {"success": False, "proof": []}
    symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    proofs = []
    for r in RULES:
        if r["then"].lower() == str(goal).lower():
            needed = r["if"]
            matched = [s for s in needed if s in symptoms]
            missing = [s for s in needed if s not in symptoms]
            proofs.append({"rule_id": r["id"], "if": needed, "matched": matched, "missing": missing, "weight": r.get("weight", 1.0)})
    if not proofs:
        return {"success": False, "proof": []}
    # best proof = fewest missing antecedents
    best = sorted(proofs, key=lambda x: len(x["missing"]))[0]
    best["success"] = (len(best["missing"]) == 0)
    return best

# ---------------------------
# 3) Build symptom->rule->disease graph and BFS/DFS
# ---------------------------
def build_rule_graph():
    graph = {}
    for r in RULES:
        rnode = "r:" + r["id"]
        dnode = "d:" + r["then"]
        graph.setdefault(rnode, []).append(dnode)
        graph.setdefault(dnode, [])
        for s in r["if"]:
            snode = "s:" + s
            graph.setdefault(snode, []).append(rnode)
            graph.setdefault(rnode, [])
    return graph

GRAPH = build_rule_graph()

def bfs_find_path(symptoms, target):
    if not target:
        return None
    starts = ["s:" + s for s in symptoms]
    target_node = "d:" + target
    from collections import deque
    q = deque()
    visited = set()
    for s in starts:
        q.append((s, [s]))
        visited.add(s)
    while q:
        node, path = q.popleft()
        if node == target_node:
            return path
        for nb in GRAPH.get(node, []):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, path + [nb]))
    return None

def dfs_find_path(symptoms, target):
    if not target:
        return None
    starts = ["s:" + s for s in symptoms]
    target_node = "d:" + target
    visited = set()
    stack = []
    for s in starts:
        stack.append((s, [s]))
    while stack:
        node, path = stack.pop()
        if node == target_node:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nb in GRAPH.get(node, []):
            stack.append((nb, path + [nb]))
    return None

# ---------------------------
# 4) Classifier predict + explain
# ---------------------------
def predict_and_explain(symptom_text):
    X = vectorizer.transform([symptom_text])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_prob = float(probs[top_idx])
    # explain: tokens present and coef contribution for the predicted class
    feat_names = vectorizer.get_feature_names_out()
    tokens = [t for t in symptom_text.lower().split() if t.strip()]
    contributions = []
    coef = clf.coef_
    class_idx = list(classes).index(top_class)
    coef_vec = coef[class_idx]
    for t in sorted(set(tokens)):
        if t in feat_names:
            idx = int(list(feat_names).index(t))
            contributions.append({"token": t, "coef": float(coef_vec[idx])})
    contributions = sorted(contributions, key=lambda x: abs(x["coef"]), reverse=True)
    return top_class, top_prob, contributions

# ---------------------------
# 5) Treatment suggestions (simple mapping)
# ---------------------------
TREATMENT_MAP = {
    "Flu": "Supportive care: rest, fluids, antipyretics; advise follow-up if worsening.",
    "Meningitis": "Urgent hospital referral; start empiric antibiotics until meningitis ruled out.",
    "Cardiac Event": "Emergency response: ECG, aspirin, oxygen; urgent cardiology referral.",
    "TB suspect": "Isolate patient; send for TB testing and start protocol if confirmed.",
    "Pharyngitis": "Analgesics; consider antibiotics if streptococcal infection suspected.",
    "Asthma Exacerbation": "Bronchodilators, inhaled steroids, oxygen if hypoxic.",
    "Gastroenteritis": "Oral rehydration, antiemetics if needed; follow-up."
}

# ---------------------------
# 6) Streamlit UI layout
# ---------------------------
st.title("AI Medical Assistant — Streamlit Demo")
st.markdown("Combines ML (classifier), rule-based reasoning (forward/backward chaining) and graph search (BFS/DFS). For demo only — not for clinical use.")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("Patient Input")
    name = st.text_input("Patient name")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Unknown", "Female", "Male", "Other"])
    symptoms_input = st.text_area("Symptoms (comma-separated)", value="fever, cough")
    extra_notes = st.text_area("Extra notes / vitals (optional)", value="", height=80)

    if st.button("Analyze & Save (optional)"):
        # parse symptoms
        symptoms_list = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]
        if len(symptoms_list) == 0:
            st.error("Please enter at least one symptom (comma separated).")
        else:
            # classifier
            top_class, top_prob, contribs = predict_and_explain(" ".join(symptoms_list))
            # forward chaining
            fwd = forward_chaining(symptoms_list)
            # backward chaining
            bwd = backward_chaining(top_class, symptoms_list)
            # BFS/DFS
            bfs_path = bfs_find_path(symptoms_list, top_class)
            dfs_path = dfs_find_path(symptoms_list, top_class)
            # combined ranking (simple)
            combined = {}
            for d, sc in fwd["scores"].items():
                combined[d] = combined.get(d, 0) + sc
            combined[top_class] = combined.get(top_class, 0) + top_prob
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

            # save patient record to DB (optional)
            try:
                save_patient(name if name else "Anonymous", int(age), gender, symptoms_input, extra_notes)
                st.success("Patient record saved.")
            except Exception as e:
                st.warning(f"Could not save patient: {e}")

            # display results
            st.subheader("ML Prediction")
            st.write(f"**{top_class}** — probability {top_prob:.2f}")
            st.write("Classifier token contributions (approx):", contribs)

            st.subheader("Forward Chaining — Rules Fired")
            st.json(fwd["fired"])

            st.subheader("Backward Chaining Proof (for top prediction)")
            st.json(bwd)

            st.subheader("Graph Traversal")
            st.write("BFS path:", bfs_path)
            st.write("DFS path:", dfs_path)

            st.subheader("Combined ranking (rules + classifier)")
            st.write(ranked)

            st.subheader("Suggested treatment (simple mapping)")
            st.write(TREATMENT_MAP.get(top_class, "Standard care; further diagnostics required."))

with right_col:
    st.header("Records & System Info")
    if st.button("Show saved patients"):
        df = list_patients()
        st.dataframe(df)
    st.markdown("**System info**")
    st.write("Rules:", len(RULES))
    st.write("Graph nodes:", len(GRAPH))
    st.write("Classifier classes:", list(clf.classes_))
    st.write("Model vectorizer features (sample):", vectorizer.get_feature_names_out()[:20].tolist())

st.markdown("---")
st.caption("This is a demo system combining ML and rule-based reasoning for teaching and demonstration. Not for clinical decision-making. For production, use proper clinical datasets, authentication, encryption, and HIPAA-compliant infrastructure.")
