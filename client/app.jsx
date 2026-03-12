import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Terminal, Play, CheckCircle, Activity, Code, ShieldAlert, Cpu, Network, Zap, GitCommit, Database, Layers } from 'lucide-react';

const API_BASE = "http://localhost:8000";

// --- Components ---

const AgentNode = ({ name, icon: Icon, isActive, isCompleted, isParallel = false }) => {
    // Styling logic
    const baseStyle = "flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-sm font-bold border-2 transition-all duration-500 w-32 relative z-10";
    let statusStyle = "bg-[#09090b] text-slate-500 border-slate-800"; // Pending
    let glow = "";

    if (isActive) {
        statusStyle = "bg-[#09090b] text-cyan-400 border-cyan-500/50 shadow-[0_0_15px_rgba(34,211,238,0.3)]";
        glow = "absolute -inset-1 bg-cyan-500/20 blur-md rounded-xl -z-10 animate-pulse";
    } else if (isCompleted) {
        statusStyle = "bg-cyan-950/30 text-cyan-600 border-cyan-900/50";
    }

    if (isParallel) {
        return (
            <div className="flex flex-col gap-2 relative">
                {glow && <div className={glow}></div>}
                {[1, 2, 3].map(i => (
                    <div key={i} className={`${baseStyle} ${statusStyle} !h-8 !w-36`}>
                        <Icon size={14} className={isActive ? "animate-spin-slow" : ""} />
                        <span>{name} {i}</span>
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div className="relative">
            {glow && <div className={glow}></div>}
            <div className={`${baseStyle} ${statusStyle}`}>
                <Icon size={16} className={isActive ? "animate-pulse" : ""} />
                <span>{name}</span>
            </div>
        </div>
    );
};

const LiveWorkflowGraph = ({ activeAgent, status }) => {
    // Determine agent states
    const states = {
        Profiler: { active: activeAgent === 'Profiler', completed: ['Scientist', 'Executor', 'Judge', 'COMPLETED'].includes(activeAgent) || status === 'COMPLETED' },
        Scientist: { active: activeAgent === 'Scientist', completed: ['Executor', 'Judge', 'COMPLETED'].includes(activeAgent) || status === 'COMPLETED' },
        Executor: { active: activeAgent === 'Executor', completed: ['Judge', 'COMPLETED'].includes(activeAgent) || status === 'COMPLETED' },
        Judge: { active: activeAgent === 'Judge', completed: status === 'COMPLETED' },
    };

    return (
        <div className="w-full bg-[#09090b]/80 backdrop-blur-xl border border-slate-800/50 rounded-2xl p-6 mb-8 flex items-center justify-between relative overflow-hidden">
            {/* Background decorative grid */}
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiLz48L3N2Zz4=')] opacity-50"></div>
            
            <div className="z-10 w-full flex items-center justify-between px-8">
                <AgentNode name="Profiler" icon={Database} isActive={states.Profiler.active} isCompleted={states.Profiler.completed} />
                <div className={`flex-1 h-0.5 mx-4 ${states.Profiler.completed ? 'bg-cyan-500/50' : 'bg-slate-800'}`}></div>
                
                <AgentNode name="Scientist" icon={Network} isActive={states.Scientist.active} isCompleted={states.Scientist.completed} />
                <div className={`flex-1 h-0.5 mx-4 ${states.Scientist.completed ? 'bg-cyan-500/50' : 'bg-slate-800'}`}></div>
                
                <AgentNode name="Executor" icon={Cpu} isActive={states.Executor.active} isCompleted={states.Executor.completed} isParallel={true} />
                <div className={`flex-1 h-0.5 mx-4 ${states.Executor.completed ? 'bg-cyan-500/50' : 'bg-slate-800'}`}></div>
                
                <AgentNode name="Judge" icon={CheckCircle} isActive={states.Judge.active} isCompleted={states.Judge.completed} />
            </div>
        </div>
    );
};


// --- Main App ---

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

    // Determines who is working right now based on who just finished
    const getNextAgent = (lastAgent) => {
        if (!lastAgent || lastAgent === "System") return "Profiler";
        if (lastAgent.toLowerCase() === "profiler") return "Scientist";
        if (lastAgent.toLowerCase() === "scientist") return "Executor";
        if (lastAgent.toLowerCase() === "executor") return "Judge";
        return "System";
    };

    const activeAgent = status === "RUNNING" 
        ? (logs.length > 0 ? getNextAgent(logs[logs.length - 1].agent) : "Profiler")
        : (status === "COMPLETED" ? "COMPLETED" : "IDLE");

    // Polling logic
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
            }, 3000);
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
            const payload = { dataset_path: datasetPath, target_column: targetCol };
            const res = await axios.post(`${API_BASE}/experiment/start`, payload);
            setExpId(res.data.experiment_id);
            setStatus("RUNNING");
            setLogs([{ agent: "System", message: "Initialising LangGraph workflow..." }]);
            setWinner(null);
        } catch (err) {
            setError("Failed to start experiment. Check server connection.");
        }
    };

    return (
        <div className="min-h-screen bg-[#050505] text-slate-300 font-sans p-6 selection:bg-cyan-500/30 selection:text-cyan-200">
            {/* Header */}
            <div className="max-w-7xl mx-auto flex justify-between items-center mb-8 border-b border-white/5 pb-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)]">
                        <Layers className="text-white" size={20} />
                    </div>
                    <div>
                        <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
                            AGNOSTOS <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">LAB</span>
                        </h1>
                        <p className="text-slate-500 text-xs font-mono tracking-widest uppercase">Parallel Convergence Matrix</p>
                    </div>
                </div>
                {status === "RUNNING" && (
                    <div className="flex items-center gap-2 bg-cyan-950/40 text-cyan-400 px-4 py-2 rounded-full border border-cyan-500/20 text-xs font-mono uppercase tracking-wider shadow-[0_0_10px_rgba(6,182,212,0.1)]">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                        </span>
                        Live: Exp. #{expId}
                    </div>
                )}
            </div>

            <div className="max-w-7xl mx-auto">
                {/* Live Graph visualizer (only show running/completed) */}
                {(status === "RUNNING" || status === "COMPLETED") && (
                    <LiveWorkflowGraph activeAgent={activeAgent} status={status} />
                )}

                <div className="grid grid-cols-12 gap-8">
                    {/* Left Column: Controls */}
                    <div className="col-span-12 lg:col-span-3 space-y-6">
                        <div className="bg-[#09090b]/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-800/50 shadow-2xl">
                            <h2 className="text-sm font-mono tracking-widest text-white uppercase flex items-center gap-2 mb-6 border-b border-slate-800 pb-3">
                                <Zap size={14} className="text-cyan-500" /> Init Protocol
                            </h2>
                            <div className="space-y-5">
                                <div>
                                    <label className="text-[10px] font-mono tracking-widest text-slate-500 uppercase mb-2 block">Data Source</label>
                                    <label className="flex flex-col items-center justify-center w-full h-28 bg-[#050505] border border-dashed border-slate-700/50 rounded-xl cursor-pointer hover:border-cyan-500/50 transition-all hover:bg-cyan-950/10 group">
                                        <input type="file" accept=".csv,.json,.parquet" className="hidden" onChange={handleFileUpload} />
                                        {uploading ? (
                                            <span className="text-cyan-400 text-xs font-mono animate-pulse">Establishing uplink...</span>
                                        ) : uploadedFileName ? (
                                            <div className="flex flex-col items-center text-center px-4">
                                                <Database size={16} className="text-cyan-500 mb-2" />
                                                <span className="text-white text-xs font-mono break-all">{uploadedFileName}</span>
                                            </div>
                                        ) : (
                                            <span className="text-slate-500 text-xs font-mono group-hover:text-cyan-500/70 transition-colors">Select Payload (.csv)</span>
                                        )}
                                    </label>
                                </div>
                                <div>
                                    <label className="text-[10px] font-mono tracking-widest text-slate-500 uppercase mb-2 block">Target Variable</label>
                                    <input
                                        type="text"
                                        value={targetCol}
                                        className="w-full bg-[#050505] border border-slate-800 rounded-xl px-4 py-3 text-sm font-mono text-cyan-50 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
                                        onChange={(e) => setTargetCol(e.target.value)}
                                    />
                                </div>
                                <button
                                    onClick={handleStartExperiment}
                                    disabled={status === "RUNNING" || !datasetPath || uploading}
                                    className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 text-white font-mono tracking-widest uppercase text-xs py-4 rounded-xl transition-all shadow-[0_0_20px_rgba(6,182,212,0.2)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)] disabled:shadow-none mt-4 relative overflow-hidden group"
                                >
                                    <span className="relative z-10 flex items-center justify-center gap-2">
                                        {status === "RUNNING" ? <Activity size={14} className="animate-spin" /> : <Play size={14} className="group-hover:translate-x-1 transition-transform" />}
                                        Initialize
                                    </span>
                                </button>
                                {error && <p className="text-red-400/90 text-[10px] font-mono mt-2 flex items-center gap-1"><ShieldAlert size={10} /> {error}</p>}
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Terminal */}
                    <div className="col-span-12 lg:col-span-9 space-y-6">
                        <div className="bg-[#050505] rounded-2xl border border-slate-800/80 shadow-2xl overflow-hidden flex flex-col h-[550px] relative">
                            {/* Terminal Top Bar */}
                            <div className="bg-[#09090b] px-4 py-3 flex items-center gap-2 border-b border-slate-800/80 z-20">
                                <Terminal size={14} className="text-cyan-500/50" />
                                <span className="text-[10px] font-mono font-bold text-slate-500 uppercase tracking-widest">Global Event Stream</span>
                            </div>

                            <div className="flex-grow flex relative">
                                {/* Main Shared Terminal (Always visible, shrinks/fades when Parallel is active, but we'll just keep it full width or half width) */}
                                <div className={`p-6 overflow-y-auto space-y-3 font-mono text-xs flex-grow transition-all duration-700 z-10 
                                    ${activeAgent === 'Executor' ? 'border-r border-slate-800/50 hidden md:block w-1/3 opacity-50 blur-[1px]' : 'w-full'}`}>
                                    {logs.length === 0 && <p className="text-slate-700 italic">Awaiting inputs...</p>}
                                    {logs.map((log, i) => (
                                        <div key={i} className="flex flex-col gap-1 border-l border-slate-800/50 pl-3 py-1">
                                            <span className="text-cyan-500/80 font-bold uppercase tracking-widest text-[9px]">[{log.agent}]</span>
                                            <span className="text-slate-400 leading-relaxed break-words">{log.message}</span>
                                        </div>
                                    ))}
                                    {status === "RUNNING" && activeAgent !== 'Executor' && (
                                        <div className="text-cyan-400 animate-pulse pl-3 mt-4 flex items-center gap-2">
                                            <span className="w-1.5 h-3 bg-cyan-400"></span>
                                            <span>
                                                {logs.length > 0 
                                                    ? `${activeAgent} routine active...` 
                                                    : "Boot sequence initiated..."}
                                            </span>
                                        </div>
                                    )}
                                </div>

                                {/* Parallel view overlays/slots when Executor is active */}
                                {status === "RUNNING" && activeAgent === 'Executor' && (
                                    <div className="flex-1 bg-[#09090b]/50 backdrop-blur-sm p-4 grid grid-cols-1 md:grid-cols-3 gap-4 animate-in fade-in duration-700 z-20">
                                        {[1, 2, 3].map(i => (
                                            <div key={i} className="bg-[#050505] border border-cyan-900/30 rounded-xl p-4 flex flex-col relative overflow-hidden">
                                                <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"></div>
                                                <div className="flex justify-between items-center mb-4">
                                                    <span className="text-[10px] font-mono text-cyan-600 uppercase tracking-widest">Tread_{i}</span>
                                                    <span className="flex h-1.5 w-1.5 translate-y-0.5 rounded-full bg-cyan-500 animate-pulse"></span>
                                                </div>
                                                <div className="flex-1 flex items-center justify-center flex-col gap-3">
                                                    <Cpu size={24} className="text-cyan-500/30 animate-pulse" />
                                                    <div className="text-[10px] font-mono text-slate-500 text-center uppercase tracking-widest">
                                                        Compiling Model <br/>
                                                        <span className="text-cyan-500/50 animate-pulse">cloud runtime</span>
                                                    </div>
                                                </div>
                                                <div className="mt-auto pt-4 border-t border-slate-800/50">
                                                    <div className="w-full bg-slate-900 rounded-full h-1 overflow-hidden">
                                                        <div className="bg-cyan-500 h-1 rounded-full animate-[loading_2s_ease-in-out_infinite] w-1/3"></div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Bottom Section: Result Reveal */}
                    {status === "COMPLETED" && winner && (
                        <div className="col-span-12 animate-in fade-in slide-in-from-bottom-8 duration-700 mt-4">
                            <div className="bg-[#09090b] border border-cyan-500/30 p-8 rounded-2xl shadow-[0_0_50px_rgba(6,182,212,0.1)] relative overflow-hidden group">
                                <div className="absolute -inset-24 bg-gradient-to-tr from-cyan-500/10 to-blue-500/10 blur-3xl opacity-50 group-hover:opacity-70 transition-opacity duration-1000"></div>
                                
                                <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
                                    <div className="col-span-12 lg:col-span-6 flex flex-col justify-center">
                                        <div className="flex items-center gap-2 mb-4">
                                            <div className="bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 w-fit px-3 py-1 rounded-full text-[10px] font-mono tracking-widest uppercase flex items-center gap-1.5">
                                                <CheckCircle size={10} /> Optimal Pattern Converted
                                            </div>
                                        </div>
                                        <h2 className="text-5xl font-black text-white tracking-tight mb-2 drop-shadow-md">{winner.approach_name}</h2>
                                        <p className="text-slate-400 text-sm mb-8 font-mono">Verified by Judge as the peak performer for local topology.</p>
                                        
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-[#050505] border border-slate-800 p-4 rounded-xl flex flex-col justify-between">
                                                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2">Accuracy</p>
                                                <p className="text-2xl font-black text-cyan-400">{winner.metrics?.accuracy ? (winner.metrics.accuracy * 100).toFixed(1) : (winner.accuracy * 100).toFixed(1)}<span className="text-sm font-normal text-cyan-600">%</span></p>
                                            </div>
                                            <div className="bg-[#050505] border border-slate-800 p-4 rounded-xl flex flex-col justify-between">
                                                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2">F1 Score</p>
                                                <p className="text-2xl font-black text-white">{winner.metrics?.f1_score ? (winner.metrics.f1_score * 100).toFixed(1) : '--'}<span className="text-sm font-normal text-slate-600">%</span></p>
                                            </div>
                                            <div className="bg-[#050505] border border-slate-800 p-4 rounded-xl flex flex-col justify-between">
                                                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2">Precision</p>
                                                <p className="text-2xl font-black text-white">{winner.metrics?.precision ? (winner.metrics.precision * 100).toFixed(1) : '--'}<span className="text-sm font-normal text-slate-600">%</span></p>
                                            </div>
                                            <div className="bg-[#050505] border border-slate-800 p-4 rounded-xl flex flex-col justify-between">
                                                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2">Recall</p>
                                                <p className="text-2xl font-black text-white">{winner.metrics?.recall ? (winner.metrics.recall * 100).toFixed(1) : '--'}<span className="text-sm font-normal text-slate-600">%</span></p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="col-span-12 lg:col-span-6">
                                        <div className="bg-[#050505] rounded-xl border border-slate-800 h-full flex flex-col">
                                            <div className="bg-[#09090b] px-4 py-2 border-b border-slate-800 flex justify-between items-center rounded-t-xl">
                                                <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                                                    <GitCommit size={10} /> source_code.py
                                                </span>
                                                <span className="flex gap-1.5">
                                                    <span className="w-2 h-2 rounded-full bg-slate-800"></span>
                                                    <span className="w-2 h-2 rounded-full bg-slate-800"></span>
                                                    <span className="w-2 h-2 rounded-full bg-slate-800"></span>
                                                </span>
                                            </div>
                                            <div className="p-4 flex-grow overflow-hidden relative">
                                                <pre className="text-[10px] text-cyan-100/70 font-mono h-48 overflow-y-auto scrollbar-hide selection:bg-cyan-900">
                                                    <code>{winner.code}</code>
                                                </pre>
                                                <div className="absolute bottom-0 left-0 w-full h-8 bg-gradient-to-t from-[#050505] to-transparent pointer-events-none"></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            
            {/* Custom animations for the progress bars */}
            <style jsx global>{`
                @keyframes loading {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(300%); }
                }
            `}</style>
        </div>
    );
}

export default App;
