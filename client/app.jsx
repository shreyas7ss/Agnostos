import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Terminal, Play, CheckCircle, Activity, Code, ShieldAlert } from 'lucide-react';

const API_BASE = "http://localhost:8000";

function App() {
    const [datasetPath, setDatasetPath] = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [uploading, setUploading] = useState(false);
    const [targetCol, setTargetCol] = useState("target");
    const [expId, setExpId] = useState(null);
    const [status, setStatus] = useState("idle");
    const [logs, setLogs] = useState([]);
    const [winner, setWinner] = useState(null);
    const [error, setError] = useState(null);

    // Polling logic to match your GET /experiments/{exp_id} route
    useEffect(() => {
        let interval;
        if (status === "RUNNING" && expId) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${API_BASE}/experiments/${expId}`);
                    setLogs(res.data.logs);

                    if (res.data.status === "COMPLETED") {
                        setStatus("COMPLETED");
                        setWinner(res.data.final_result);
                        clearInterval(interval);
                    }
                } catch (err) {
                    console.error("Polling error:", err);
                }
            }, 3000); // Poll every 3 seconds
        }
        return () => clearInterval(interval);
    }, [status, expId]);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setUploading(true);
        setUploadedFileName(file.name);
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await axios.post(`${API_BASE}/upload`, formData);
            setDatasetPath(res.data.dataset_path);
        } catch (err) {
            setError("File upload failed. Check server connection.");
            setUploadedFileName("");
        } finally {
            setUploading(false);
        }
    };

    const handleStartExperiment = async () => {
        setError(null);
        try {
            // Matches your ExperimentRequest Pydantic model
            const payload = {
                dataset_path: datasetPath,
                target_column: targetCol
            };

            const res = await axios.post(`${API_BASE}/experiment/start`, payload);
            setExpId(res.data.experiment_id);
            setStatus("RUNNING");
            setLogs([{ agent: "System", message: "Initialising LangGraph workflow..." }]);
        } catch (err) {
            setError("Failed to start experiment. Check server connection.");
        }
    };

    return (
        <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6">
            {/* Header */}
            <div className="max-w-6xl mx-auto flex justify-between items-center mb-10 border-b border-slate-800 pb-6">
                <div>
                    <h1 className="text-3xl font-black tracking-tighter text-white">AGNOSTOS <span className="text-blue-500">LAB</span></h1>
                    <p className="text-slate-500 text-sm">v1.0.0 • Parallel Agentic Training</p>
                </div>
                {status === "RUNNING" && (
                    <div className="flex items-center gap-2 bg-blue-500/10 text-blue-400 px-4 py-1.5 rounded-full border border-blue-500/20 text-sm">
                        <Activity size={14} className="animate-spin" /> Live Experiment #{expId}
                    </div>
                )}
            </div>

            <div className="max-w-6xl mx-auto grid grid-cols-12 gap-6">

                {/* Left Column: Controls */}
                <div className="col-span-12 lg:col-span-4 space-y-6">
                    <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-800 shadow-xl">
                        <h2 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
                            <Play size={18} className="text-blue-500" /> Configuration
                        </h2>
                        <div className="space-y-4">
                            <div>
                                <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Dataset File</label>
                                <label className="flex flex-col items-center justify-center w-full h-24 bg-[#0f172a] border-2 border-dashed border-slate-700 rounded-lg cursor-pointer hover:border-blue-500 transition-colors">
                                    <input type="file" accept=".csv,.json,.parquet" className="hidden" onChange={handleFileUpload} />
                                    {uploading ? (
                                        <span className="text-blue-400 text-sm animate-pulse">Uploading...</span>
                                    ) : uploadedFileName ? (
                                        <span className="text-green-400 text-sm font-mono">✓ {uploadedFileName}</span>
                                    ) : (
                                        <span className="text-slate-500 text-sm">Click to upload .csv / .json / .parquet</span>
                                    )}
                                </label>
                            </div>
                            <div>
                                <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Target Column</label>
                                <input
                                    type="text"
                                    value={targetCol}
                                    className="w-full bg-[#0f172a] border border-slate-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
                                    onChange={(e) => setTargetCol(e.target.value)}
                                />
                            </div>
                            <button
                                onClick={handleStartExperiment}
                                disabled={status === "RUNNING" || !datasetPath || uploading}
                                className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-bold py-3 rounded-lg transition-all shadow-lg shadow-blue-900/20 mt-2"
                            >
                                Execute Pipeline
                            </button>
                            {error && <p className="text-red-400 text-xs mt-2 flex items-center gap-1"><ShieldAlert size={12} /> {error}</p>}
                        </div>
                    </div>
                </div>

                {/* Right Column: Terminal */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                    <div className="bg-black rounded-2xl border border-slate-800 shadow-2xl overflow-hidden flex flex-col h-[500px]">
                        <div className="bg-[#1e293b] px-4 py-2 flex items-center gap-2 border-b border-slate-800">
                            <Terminal size={14} className="text-slate-400" />
                            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Agent Logs</span>
                        </div>
                        <div className="p-4 overflow-y-auto space-y-2 font-mono text-sm flex-grow">
                            {logs.length === 0 && <p className="text-slate-700 italic">Waiting for execution instructions...</p>}
                            {logs.map((log, i) => (
                                <div key={i} className="flex gap-3 border-l-2 border-slate-800 pl-3 py-1">
                                    <span className="text-blue-400 font-bold min-w-[80px] text-xs">[{log.agent}]</span>
                                    <span className="text-slate-300">{log.message}</span>
                                </div>
                            ))}
                            {status === "RUNNING" && <div className="text-blue-500 animate-pulse pl-3">▋ Agent is thinking...</div>}
                        </div>
                    </div>
                </div>

                {/* Bottom Section: Result Reveal */}
                {status === "COMPLETED" && winner && (
                    <div className="col-span-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="bg-gradient-to-br from-blue-600 to-indigo-700 p-8 rounded-3xl shadow-2xl relative overflow-hidden">
                            <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div>
                                    <div className="bg-white/10 w-fit px-3 py-1 rounded-full text-xs font-bold mb-4">🏆 BEST MODEL SELECTED</div>
                                    <h2 className="text-4xl font-black text-white mb-2">{winner.approach_name}</h2>
                                    <p className="text-blue-100 mb-6">The Judge has verified this model as the most optimal solution for your data profile.</p>
                                    <div className="flex gap-4">
                                        <div className="bg-black/20 p-4 rounded-2xl backdrop-blur-sm">
                                            <p className="text-xs font-bold text-blue-200 uppercase">Accuracy</p>
                                            <p className="text-3xl font-black text-white">{(winner.accuracy * 100).toFixed(1)}%</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="bg-black/40 rounded-2xl p-4 border border-white/10">
                                    <div className="flex justify-between mb-2">
                                        <span className="text-xs font-bold text-blue-200">SOURCE CODE</span>
                                        <Code size={14} />
                                    </div>
                                    <pre className="text-[10px] text-blue-100 overflow-x-auto h-40 font-mono scrollbar-hide">
                                        <code>{winner.code}</code>
                                    </pre>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
