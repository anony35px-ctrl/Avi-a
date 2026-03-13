# ==========================================================
# PROYECTO: CLASIFICADOR DE SPAM CON NAIVE BAYES
# Autor: Gael Emiliano Mejia Alegria
# Este notebook entrena un modelo para distinguir mensajes
# SPAM de HAM usando TF-IDF y Naive Bayes.
# ==========================================================

# ==========================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib  # Para guardar el modelo

# ==========================================================
# 2. CREACIÓN DEL DATASET (30 SPAM + 30 HAM)
# ==========================================================
spam_texts = [
    "Congratulations! You have been selected to win a brand new iPhone. Click here to claim your prize now.",
    "Limited time offer! Get 80% discount on all products if you buy today.",
    "You are the lucky winner of a $1000 gift card. Claim it before it expires.",
    "Earn money from home with this simple method. Start making $500 a day.",
    "Your account has been temporarily suspended. Verify your information immediately.",
    "Special promotion just for you. Buy one get one free only today.",
    "Hot singles are waiting to meet you in your area. Click here now.",
    "You have an unclaimed reward. Log in now to receive your prize.",
    "Act fast! This exclusive investment opportunity guarantees huge profits.",
    "Congratulations! Your email has won our international lottery.",
    "Your payment was declined. Update your billing information now.",
    "Last chance to claim your free vacation tickets. Offer ends tonight.",
    "Make money online with no experience required. Join now.",
    "You have been selected for a free trial of our premium service.",
    "Your package delivery failed. Click here to reschedule immediately.",
    "Get rich quickly using this proven system used by thousands.",
    "Important notice: your bank account requires verification.",
    "You are pre-approved for a personal loan up to $10,000.",
    "Download this app now and receive free rewards instantly.",
    "Only a few spots left. Register now before it's too late.",
    "Your profile has been chosen for a special bonus reward.",
    "Click here to unlock your exclusive membership benefits.",
    "You have received a confidential financial proposal.",
    "Earn passive income every day with this investment strategy.",
    "Your subscription will expire soon. Renew now to avoid interruption.",
    "Win big prizes by entering our online contest today.",
    "Exclusive deal: save 70% on luxury watches today.",
    "Security alert! Someone tried to access your account.",
    "You have been randomly selected for a special survey reward.",
    "Final notice: claim your bonus before midnight tonight."
]

ham_texts = [
    "Hey, are we still meeting after school today?",
    "Don't forget to bring the homework tomorrow.",
    "I sent you the document by email, check if you received it.",
    "Let me know when you arrive home safely.",
    "Are you available to study together this weekend?",
    "Thanks for helping me with the assignment yesterday.",
    "I will call you later when I finish my class.",
    "Can you send me the notes from today's lecture?",
    "We should start working on the project soon.",
    "I'll be a little late to the meeting today.",
    "Did you understand the last math problem?",
    "Let's go eat something after the exam.",
    "I just finished the report, I'll send it to you now.",
    "Remember we have a presentation tomorrow morning.",
    "I think the teacher changed the deadline.",
    "Text me when you get to the bus station.",
    "I'm already at the library waiting for you.",
    "Do you want to practice English later?",
    "I will bring my laptop so we can finish the work.",
    "The class today was actually very interesting.",
    "Let me know if you need help with the code.",
    "I just uploaded the files to the shared folder.",
    "Can we move the meeting to tomorrow?",
    "I'm going to the store, do you need anything?",
    "I'll send you the pictures from the trip later.",
    "Don't worry, we'll figure out the solution together.",
    "I'm reviewing the material for tomorrow's test.",
    "Tell me if the program runs correctly on your computer.",
    "Let's schedule another session to finish the project.",
    "See you tomorrow morning at school."
]

# Combinar y etiquetar
all_messages = spam_texts + ham_texts
labels = ["spam"] * len(spam_texts) + ["ham"] * len(ham_texts)

data = pd.DataFrame({
    "texto": all_messages,
    "clase": labels
})

# ==========================================================
# 3. LIMPIEZA DE TEXTO
# ==========================================================
def clean_text(text):
    """
    Convierte a minúsculas, elimina URLs, menciones, números,
    caracteres especiales y espacios múltiples.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # URLs
    text = re.sub(r'@\w+', '', text)             # menciones
    text = re.sub(r'\d+', '', text)              # números
    text = re.sub(r'[^a-z\s]', '', text)         # caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()     # espacios múltiples
    return text

data["texto_limpio"] = data["texto"].apply(clean_text)

# ==========================================================
# 4. TOKENIZACIÓN Y ELIMINACIÓN DE STOPWORDS
# ==========================================================
nltk.download('stopwords')
# Incluimos stopwords en español e inglés para mayor cobertura
stop_words = set(stopwords.words("spanish") + stopwords.words("english"))

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)

data["texto_procesado"] = data["texto_limpio"].apply(remove_stopwords)

# ==========================================================
# 5. VECTORIZACIÓN TF-IDF
# ==========================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["texto_procesado"])
y = data["clase"]

# ==========================================================
# 6. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123  # semilla diferente
)

# ==========================================================
# 7. ENTRENAMIENTO DEL MODELO NAIVE BAYES
# ==========================================================
model = MultinomialNB()
model.fit(X_train, y_train)

# ==========================================================
# 8. EVALUACIÓN DEL MODELO
# ==========================================================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Matriz de confusión:")
print(cm)
print(f"Precisión del modelo: {accuracy:.4f}")

# ==========================================================
# 9. GUARDAR MODELO Y VECTORIZADOR PARA USO FUTURO
# ==========================================================
joblib.dump(model, "modelo_spam_nb.pkl")
joblib.dump(vectorizer, "vectorizer_tfidf.pkl")
print("Modelo y vectorizador guardados en disco.")

# ==========================================================
# 10. PREDICCIÓN EN TIEMPO REAL
# ==========================================================
new_message = ["Congratulations you won a free prize click now"]
print("\nMensaje a evaluar:", new_message[0])

# Aplicar el mismo preprocesamiento
cleaned = clean_text(new_message[0])
processed = remove_stopwords(cleaned)
vec = vectorizer.transform([processed])

pred = model.predict(vec)[0]
proba = model.predict_proba(vec)[0]
confidence = max(proba) * 100

print("Predicción:", pred)
print(f"Confianza: {confidence:.2f}%")