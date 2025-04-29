import React from "react";
import { useNavigate } from "react-router-dom";
import { SidebarData } from "./SidebarData";
import AddIcon from "@mui/icons-material/Add";
import "../style.css";
import { Button } from "@mui/material";

function Sidebar({ chats, onNewChat, onChatSelect }) {
  const navigate = useNavigate();

  return (
    <div className="Sidebar">
      <ul className="SidebarList">
        {SidebarData.map((val, key) => {
          const IconComponent = val.icon;
          return (
            <li
              key={key}
              className="row"
              id={window.location.pathname === val.link ? "active" : ""}
              onClick={() => navigate(val.link)}
            >
              <div className="sidebar-icon">
                {React.isValidElement(IconComponent) ? IconComponent : null}
              </div>
              <div className="sidebar-title">{val.title}</div>
            </li>
          );
        })}
      </ul>

      <div className="Newchat">
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<AddIcon />} 
          fullWidth 
          onClick={onNewChat}
        >
          New Chat
        </Button>
      </div>

      <div className="chat-history">
        <h4>Chat History</h4>
        <ul className="history-list">
          {chats.length === 0 ? (
            <p className="history-name">No history available.</p>
          ) : (
            chats.map((chat, index) => (
              <li 
                key={chat.id || index}
                onClick={() => onChatSelect(chat)}
              >
                {chat.file?.name || "Untitled Chat"}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}

export default Sidebar;