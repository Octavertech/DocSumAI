import React, { useState, useEffect } from "react";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import "bootstrap/dist/css/bootstrap.min.css";
import "../style.css";
import axios from "axios";

import { Document, Packer, Paragraph, TextRun } from "docx";
import { saveAs } from "file-saver";
import { marked } from "marked";  // Importing marked library

const Summarizer = ({ addToHistory, resetChat, currentChat }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [summary, setSummary] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showSummaryMsg, setShowSummaryMsg] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gemini");

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  useEffect(() => {
    if (currentChat?.file) {
      setSelectedFile(currentChat.file);
      setSummary(formatSummary(currentChat.summary || ""));
      setShowSummaryMsg(true);
      setError("");
      setFileUrl(URL.createObjectURL(currentChat.file));
    } else {
      resetFields();
    }
  }, [currentChat]);

  useEffect(() => {
    if (resetChat) resetFields();
  }, [resetChat]);

  const resetFields = () => {
    setSelectedFile(null);
    setFileUrl(null);
    setSummary([]);
    setError("");
    setShowSummaryMsg(false);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"].includes(file.type)) {
      setError("Only PDF, text, and DOCX files are supported.");
      return;
    }

    setSelectedFile(file);
    setError("");
    setSummary([]);
    setShowSummaryMsg(false);
    setFileUrl(URL.createObjectURL(file));
  };

  const formatSummary = (text) => {
    return text.split("\n").filter((point) => point.trim() !== "");
  };

  const handleSummarize = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    setLoading(true);
    setError("");
    setSummary([]);
    setShowSummaryMsg(false);

    // Hardcoded response
    // const response = {
    //   data: {
    //     summary: [
    //       "This study examines how Danish researchers use generative AI (GenAI) and perceive its impact on research integrity. A 2024 survey (2,534 responses) assessed 32 GenAI use cases across five research phases. Findings show:",
    //       "Three perception clusters: GenAI as a *work horse*, *language assistant*, or *research accelerator*.",
    //       "Mixed integrity views: Positive for language editing/data analysis, controversial for image/synthetic data use.",
    //       "Higher usage among junior researchers and technical fields, with no major gender differences.",
    //       "The study highlights ethical concerns and varied acceptance of GenAI in research.",
    //       "Keywords: Generative AI, research integrity, academic use cases."
    //     ],
    //   },
    // };
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    const response = await axios.post("http://127.0.0.1:8000/summarize", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

    // Using the hardcoded response
    const formattedSummary = formatSummary(response.data.summary.join("\n"));
    setSummary(formattedSummary);
    setShowSummaryMsg(true);
    addToHistory(selectedFile, response.data.summary.join("\n"));
    setLoading(false);
  };

  // MARKDOWN TO DOCX PARSER
  const parseMarkdownToDocxParagraphs = (lines) => {
    return lines.map((line) => {
      if (line.startsWith("###")) {
        return new Paragraph({
          children: [new TextRun({ text: line.replace(/^###/, "").trim(), bold: true, size: 28 })],
        });
      } else if (line.startsWith("##")) {
        return new Paragraph({
          children: [new TextRun({ text: line.replace(/^##/, "").trim(), bold: true, size: 32 })],
        });
      } else if (line.startsWith("#")) {
        return new Paragraph({
          children: [new TextRun({ text: line.replace(/^#/, "").trim(), bold: true, size: 36 })],
        });
      } else {
        return new Paragraph({
          children: line.split(/\*\*(.*?)\*\*/g).map((part, i) =>
            i % 2 === 1 ? new TextRun({ text: part, bold: true }) : new TextRun(part)
          ),
        });
      }
    });
  };

  const handleDownload = async () => {
    const doc = new Document({
      sections: [
        {
          children: parseMarkdownToDocxParagraphs(summary),
        },
      ],
    });

    const blob = await Packer.toBlob(doc);
    saveAs(blob, "FormattedSummary.docx");
  };

  // Convert markdown summary to HTML using marked
  const markdownContent = marked(summary.join("\n\n"));

  return (
    <div className="container text-center">
      <div className="button-container">
        {!selectedFile && (
          <label className="btn btn-primary">
            <AttachFileIcon />
            <input type="file" onChange={handleFileChange} accept=".pdf,.txt,.docx" hidden />
            Choose File
          </label>
        )}

        <select className="form-select mx-3" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
          <option value="gemini">Google Gemini</option>
        </select>

        {!showSummaryMsg ? (
          <button className="btn btn-success mx-3" onClick={handleSummarize} disabled={!selectedFile || loading}>
            {loading ? (
              <>
                <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                Processing...
              </>
            ) : (
              "Summarize Document"
            )}
          </button>
        ) : ''}
      </div>
      
      {fileUrl && (
        <div className="file-preview mt-4 mb-4">
          <h4>File Preview</h4>
          <iframe
            src={fileUrl}
            width="100%"
            height="600px"
            className="pdf-frame"
            title="File Preview"
          ></iframe>
        </div>
      )}

      {error && <div className="alert alert-danger mt-3">{error}</div>}

      {summary.length > 0 && (
        <div className="summary-container mt-4">
          <h4>Summary</h4>
          <div
            className="summary-content"
            dangerouslySetInnerHTML={{ __html: markdownContent }} // Render the converted HTML here
          ></div>

          <button className="btn btn-outline-primary mt-3" onClick={handleDownload}>
            Download Summary
          </button>
        </div>
      )}
    </div>
  );
};

export default Summarizer;
