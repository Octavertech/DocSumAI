import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Summarizer from "./components/Summarizer";
import Header from "./components/Header/Header";
import "./style.css";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [chats, setChats] = useState([]);
  const [currentChat, setCurrentChat] = useState({ id: Date.now(), file: null, summary: "" });
  const [resetChat, setResetChat] = useState(false);

  useEffect(() => {
    if (!currentChat) {
      handleNewChat();
    }
  }, []);

  const handleNewChat = () => {
    if (currentChat?.summary || currentChat?.file) {
      setChats((prevChats) => [...prevChats, currentChat]);
    }
    setCurrentChat({ id: Date.now(), file: null, summary: "" });
    setResetChat(true);
  };

  const addToHistory = (file, summary) => {
    setCurrentChat(prev => ({
      ...prev,
      file,
      summary: typeof summary === 'string' ? summary : JSON.stringify(summary)
    }));
    setResetChat(false);
  };

  const handleChatSelect = (chat) => {
    setCurrentChat(chat);
    setResetChat(false);
  };

  return (
    <Router>
      <div className="app-container">
        <Header />
        <Sidebar 
          chats={chats} 
          onNewChat={handleNewChat} 
          onChatSelect={handleChatSelect} 
        />
        <div className="content">
          <Routes>
            <Route 
              path="/" 
              element={
                <Summarizer 
                  currentChat={currentChat} 
                  addToHistory={addToHistory} 
                  resetChat={resetChat}
                />
              } 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;