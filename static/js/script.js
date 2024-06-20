import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI('AIzaSyBExK1-_ak4wvBuatNH0307N6Th9YfPvRc'); // Replace with your actual API key

const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const {
    GoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
} = require("@google/generative-ai");


let conversationContext = [];

async function runModel(prompt) {
    const model = genAI.getGenerativeModel({ model: "tunedModels/laxia-o51q9ymko9rr" });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = await response.text();
    return text;
}

function appendMessageToChat(message, className) {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    if (className === "outgoing") {
        chatLi.innerHTML = `<p>${message}</p>`;
    } else {
        chatLi.innerHTML = `<span class="material-symbols-outlined">smart_toy</span><p>${message}</p>`;
    }
    chatbox.appendChild(chatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
}


async function handleUserMessage(userMessage) {
    appendMessageToChat(userMessage, "outgoing");
    const prompt = conversationContext.join(" ") + userMessage;
    const answer = await runModel(prompt);
    appendMessageToChat(answer, "incoming");
    conversationContext.push(userMessage);
}

function handleChat() {
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;
    chatInput.value = "";

    handleUserMessage(userMessage);
}

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));

// Call startChatSession function to initialize the chat session (assuming triggered by an event)
startChatSession(); // This line is added to initiate the chat session

async function startChatSession() {
    const initialPrompt = "you are a humble person and have your prompts in 4 line maximum"; // Append the prompt here
    chat = model.startChat({ history: [initialPrompt] }); // Initialize chat history with the prompt
}
