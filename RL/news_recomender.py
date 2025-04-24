import gradio as gr
import requests
import numpy as np
import tensorflow as tf
import random
from collections import deque

NEWS_API_KEY = "83b897d309334daeacda19673faa0ab0"

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
agents = {cat: None for cat in categories}
headlines = {}
articles_data = {}
current_state = {}
last_category = None
last_action = {}

def fetch_headlines(category):
    global headlines, articles_data, agents, current_state, last_category

    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&pageSize=10&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    articles_data[category] = {}
    headlines[category] = []

    for article in data.get("articles", []):
        title = article["title"]
        url = article.get("url")
        preview = article.get("content") or article.get("description") or "No preview."
        articles_data[category][title] = f"{preview}\n\n🔗 [Read more]({url})"
        headlines[category].append(title)

    if agents[category] is None:
        agents[category] = DQNAgent(state_size=len(headlines[category]), action_size=len(headlines[category]))

    state = np.ones(len(headlines[category]))
    current_state[category] = state
    last_category = category
    return headlines[category]

def recommend_news(category):
    fetch_headlines(category)
    state = current_state[category]
    action_index = agents[category].act(state)
    last_action[category] = action_index
    selected_headline = headlines[category][action_index]
    return gr.update(choices=headlines[category], value=selected_headline), articles_data[category][selected_headline]

def give_feedback(headline, feedback):  
    category = last_category
    index = headlines[category].index(headline)
    state = current_state[category]
    next_state = np.ones(len(headlines[category]))  
    agents[category].remember(state, index, feedback, next_state, done=True)
    agents[category].replay()
    return f"Feedback received: {'Liked' if feedback == 1 else 'Disliked'}"

def on_headline_click(headline):
    return articles_data[last_category][headline]

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Personalized News Recommender")

    category = gr.Dropdown(choices=categories, label="Choose your interest", value="technology")
    refresh_btn = gr.Button("🔁 Recommend News")
    headline_radio = gr.Radio(choices=[], label="Click a headline to view full article")
    article_box = gr.Textbox(lines=8, label="Article Preview")
    
    like_btn = gr.Button("👍 Like")
    dislike_btn = gr.Button("👎 Dislike")
    feedback_out = gr.Textbox(label="Feedback Log", interactive=False)

    refresh_btn.click(fn=recommend_news, inputs=category, outputs=[headline_radio, article_box])
    headline_radio.change(fn=on_headline_click, inputs=headline_radio, outputs=article_box)
    like_btn.click(fn=lambda h: give_feedback(h, 1), inputs=headline_radio, outputs=feedback_out)
    dislike_btn.click(fn=lambda h: give_feedback(h, -1), inputs=headline_radio, outputs=feedback_out)

demo.launch()