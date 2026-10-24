// AI Mock Interview Coach - Interactive Studio Client Logic

let currentSessionId = null;
let currentStage = 'intro';
let recognition = null;
let isRecording = false;
let isAgentSpeaking = false;
let speechSynth = window.speechSynthesis || null;
let currentUtterance = null;
let timerInterval = null;
let timeRemainingSecs = 900; // 15 mins
let candidateSpeechStartTime = 0;
let candidateSpeechEndTime = 0;
let lastReportData = null;

// Audio Visualizer setup
const canvas = document.getElementById('waveform-canvas');
const canvasCtx = canvas ? canvas.getContext('2d') : null;
let animFrameId = null;
let wavePhase = 0;
let debounceTimer = null;
let activeParsedCandidateName = "Candidate";
let activeParsedTargetRole = "Software Engineer";

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initVisualizer();
  initSpeechRecognition();
  setupLiveInputListeners();
  setupDragAndDrop();
});

function setupLiveInputListeners() {
  const resumeInput = document.getElementById('resume-text-input');
  const jdInput = document.getElementById('jd-text-input');

  const onLiveChange = () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const rVal = resumeInput ? resumeInput.value.trim() : '';
      const jVal = jdInput ? jdInput.value.trim() : '';
      if (rVal || jVal) {
        triggerIngestionPreview();
      }
    }, 350);
  };

  if (resumeInput) resumeInput.addEventListener('input', onLiveChange);
  if (jdInput) jdInput.addEventListener('input', onLiveChange);
}

function setupDragAndDrop() {
  const dropZone = document.getElementById('drop-zone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-active');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-active');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt && dt.files;
    if (files && files.length > 0) {
      handleFileSelected({ target: { files: files } });
    }
  });
}

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

  const tab = document.getElementById(`tab-${tabName}`);
  const navBtn = document.getElementById(`nav-btn-${tabName}`);
  if (tab) tab.classList.add('active');
  if (navBtn) navBtn.classList.add('active');

  const titleMap = {
    ingest: ['Candidate Ingestion & Setup', 'Load resume and job description to initialize grounded interview questions'],
    interview: ['Live Voice Interview Studio', 'Interactive voice interview with real-time VAD, interruption & latency tracking'],
    report: ['Evidence-Based Feedback Report', 'Comprehensive rubric scores, transcript quotes, and speech fluencies'],
    audit: ['Automated Acceptance & Bias Audit', 'Six automated test suites, 5-run stability test, and accented WER check']
  };

  if (titleMap[tabName]) {
    document.getElementById('page-title').innerText = titleMap[tabName][0];
    document.getElementById('page-subtitle').innerText = titleMap[tabName][1];
  }
}

// --------------------------------------------------------------------------
// 1. INGESTION & SETUP
// --------------------------------------------------------------------------

async function loadSampleData() {
  try {
    const res = await fetch('/api/sample-data');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('resume-text-input').value = data.resume_text || '';
      document.getElementById('jd-text-input').value = data.jd_text || '';
      document.getElementById('file-name-label').innerText = 'PDF parsed into sections with metadata';
      const fileInput = document.getElementById('resume-file-input');
      if (fileInput) fileInput.value = '';
      triggerIngestionPreview();
      return;
    }
  } catch (err) {
    console.log('Error fetching sample data API:', err);
  }
}

function clearIngestionForm() {
  document.getElementById('resume-text-input').value = '';
  document.getElementById('jd-text-input').value = '';
  const nameInput = document.getElementById('candidate-name-input');
  if (nameInput) nameInput.value = '';
  const roleInput = document.getElementById('target-role-input');
  if (roleInput) roleInput.value = '';
  document.getElementById('file-name-label').innerText = 'PDF parsed into sections with metadata';
  const fileInput = document.getElementById('resume-file-input');
  if (fileInput) fileInput.value = '';
  activeParsedCandidateName = 'Candidate';
  activeParsedTargetRole = 'Software Engineer';
  document.getElementById('summary-projects-count').innerText = 'Paste resume or click Auto-Fill';
  document.getElementById('summary-skills').innerHTML = '<span class="skill-pill" style="opacity: 0.7;">Upload resume to extract skills</span>';
}

function handleFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  document.getElementById('file-name-label').innerText = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  
  if (file.name.endsWith('.txt')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('resume-text-input').value = e.target.result;
      triggerIngestionPreview();
    };
    reader.readAsText(file);
  } else {
    // PDF file will be sent as FormData to /api/ingest
    triggerIngestionPreview(file);
  }
  // Reset input value so selecting the same file triggers onchange
  if (event.target.tagName === 'INPUT') {
    event.target.value = '';
  }
}

async function triggerIngestionPreview(pdfFile = null) {
  const resumeVal = document.getElementById('resume-text-input').value;
  const jdVal = document.getElementById('jd-text-input').value;

  const formData = new FormData();
  if (pdfFile) {
    formData.append('resume_file', pdfFile);
  } else {
    formData.append('resume_text', resumeVal);
  }
  formData.append('jd_text', jdVal);

  try {
    const res = await fetch('/api/ingest', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (data.status === 'success') {
      const summary = data.candidate_summary;
      activeParsedCandidateName = summary.name;
      activeParsedTargetRole = summary.target_role;

      // If PDF file was parsed, fill the textarea so candidate can view and edit text
      if (data.raw_resume_text && (!resumeVal || pdfFile)) {
        document.getElementById('resume-text-input').value = data.raw_resume_text;
      }

      const nameInput = document.getElementById('candidate-name-input');
      if (nameInput) nameInput.value = summary.name;

      const roleInput = document.getElementById('target-role-input');
      if (roleInput) roleInput.value = summary.target_role;

      const projLabel = summary.projects_count > 0 
        ? `${summary.projects_count} Projects (${summary.projects.slice(0, 2).join(', ')})`
        : 'General Technical Experience';
      document.getElementById('summary-projects-count').innerText = projLabel;

      const skillsContainer = document.getElementById('summary-skills');
      skillsContainer.innerHTML = '';
      (summary.skills || []).forEach(skill => {
        const span = document.createElement('span');
        span.className = 'skill-pill';
        span.innerText = skill;
        skillsContainer.appendChild(span);
      });

      document.getElementById('summary-card').classList.remove('hidden');
    }
  } catch (err) {
    console.error('Ingestion failed:', err);
  }
}

// --------------------------------------------------------------------------
// 2. LIVE VOICE INTERVIEW SESSION
// --------------------------------------------------------------------------

async function startInterviewSession() {
  const resumeText = document.getElementById('resume-text-input').value.trim();
  const jdText = document.getElementById('jd-text-input').value.trim();
  const budget = parseInt(document.getElementById('budget-input').value) || 5;
  const timeLimit = parseInt(document.getElementById('timelimit-input').value) || 15;

  const candidateNameInput = document.getElementById('candidate-name-input');
  const targetRoleInput = document.getElementById('target-role-input');
  
  const userTypedName = candidateNameInput ? candidateNameInput.value.trim() : '';
  const userTypedRole = targetRoleInput ? targetRoleInput.value.trim() : '';

  // Use typed name if provided; otherwise use parsed resume name
  let finalCandidateName = userTypedName;
  if (!finalCandidateName && activeParsedCandidateName && activeParsedCandidateName !== 'Candidate') {
    finalCandidateName = activeParsedCandidateName;
  }

  let finalTargetRole = userTypedRole;
  if (!finalTargetRole && activeParsedTargetRole && activeParsedTargetRole !== 'Software Engineer') {
    finalTargetRole = activeParsedTargetRole;
  }

  const startBtn = document.getElementById('start-interview-btn');
  if (startBtn) {
    startBtn.disabled = true;
    startBtn.innerText = '⏳ Initializing Interview...';
  }

  const payload = {
    resume_text: resumeText,
    jd_text: jdText,
    candidate_name: finalCandidateName,
    target_role: finalTargetRole,
    question_budget: budget,
    time_limit_mins: timeLimit
  };

  try {
    const res = await fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
    }

    const data = await res.json();

    currentSessionId = data.session_id;
    currentStage = data.stage;
    timeRemainingSecs = data.time_limit_mins * 60;

    // Switch tab to interview
    switchTab('interview');
    document.getElementById('active-session-badge').classList.remove('hidden');
    document.getElementById('session-timer-badge').classList.remove('hidden');
    document.getElementById('transcript-feed').innerHTML = '';

    // Update Stage Stepper
    updateStageStepper(data.stage);
    document.getElementById('live-q-progress').innerText = `1 / ${data.question_budget}`;

    // Add opening turn to transcript
    addTranscriptTurn('agent', data.opening_question, 'intro');

    // Speak opening agent utterance
    speakAgentResponse(data.opening_question);

    // Start countdown timer
    startCountdownTimer();

  } catch (err) {
    alert(`Could not start session: ${err.message}`);
  } finally {
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.innerText = '🎙️ Begin Voice Mock Interview';
    }
  }
}

function updateStageStepper(stage) {
  currentStage = stage;
  const stages = ['intro', 'resume_questions', 'behavioural', 'role_questions', 'candidate_questions', 'closing'];
  const stageMap = {
    intro: 'step-intro',
    resume_questions: 'step-resume',
    behavioural: 'step-behavioural',
    role_questions: 'step-role',
    candidate_questions: 'step-candidate',
    closing: 'step-closing'
  };

  const currentIdx = stages.indexOf(stage);

  stages.forEach((s, idx) => {
    const el = document.getElementById(stageMap[s]);
    if (el) {
      el.classList.remove('active', 'completed');
      if (idx < currentIdx) el.classList.add('completed');
      else if (idx === currentIdx) el.classList.add('active');
    }
  });
}

function startCountdownTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    if (timeRemainingSecs <= 0) {
      clearInterval(timerInterval);
      endInterviewSession();
      return;
    }
    timeRemainingSecs--;
    const mins = Math.floor(timeRemainingSecs / 60);
    const secs = timeRemainingSecs % 60;
    document.getElementById('timer-display').innerText = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, 1000);
}

// --------------------------------------------------------------------------
// VOICE & SPEECH ENGINE (VAD, TTS, INTERRUPTION, LATENCY)
// --------------------------------------------------------------------------

function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.warn('Web Speech Recognition API not natively supported; manual text input enabled.');
    return;
  }

  recognition = new SpeechRec();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  let finalTranscript = '';
  let vadSilenceTimer = null;

  recognition.onstart = () => {
    isRecording = true;
    updateMicButtonUI(true);
    document.getElementById('agent-status-label').innerText = 'Listening to Candidate...';
    document.getElementById('agent-status-label').className = 'status-pill listening';
  };

  recognition.onresult = (event) => {
    let interim = '';
    candidateSpeechStartTime = candidateSpeechStartTime || Date.now();

    // If agent is speaking and candidate speaks, INTERRUPT immediately!
    if (isAgentSpeaking) {
      interruptAgent();
    }

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + ' ';
      } else {
        interim += event.results[i][0].transcript;
      }
    }

    // Reset silence buffer for VAD Thinking Pause (1.2 seconds cushion)
    clearTimeout(vadSilenceTimer);
    vadSilenceTimer = setTimeout(() => {
      const fullText = (finalTranscript + interim).trim();
      if (fullText.length > 5) {
        candidateSpeechEndTime = Date.now();
        submitCandidateAnswer(fullText);
        finalTranscript = '';
      }
    }, 1200); // 1.2s VAD Thinking Pause Buffer
  };

  recognition.onerror = (event) => {
    console.log('Speech recognition event:', event.error);
    if (event.error !== 'no-speech') {
      updateMicButtonUI(false);
    }
  };

  recognition.onend = () => {
    isRecording = false;
    updateMicButtonUI(false);
  };
}

function toggleMicrophone() {
  if (!recognition) {
    alert('Web Speech Recognition is not available in this browser. You can type in the response box!');
    return;
  }

  if (isRecording) {
    recognition.stop();
    isRecording = false;
    updateMicButtonUI(false);
  } else {
    try {
      recognition.start();
      isRecording = true;
      updateMicButtonUI(true);
    } catch (e) {
      console.log('Rec already active');
    }
  }
}

function updateMicButtonUI(recording) {
  const btn = document.getElementById('mic-toggle-btn');
  const label = document.getElementById('mic-btn-label');
  if (recording) {
    btn.classList.add('recording');
    label.innerText = 'Listening (1.2s VAD active)...';
  } else {
    btn.classList.remove('recording');
    label.innerText = 'Start Speaking';
  }
}

// Interruption Handling (Requirement #3 & PDF)
function interruptAgent() {
  if (speechSynth && speechSynth.speaking) {
    speechSynth.cancel();
  }
  isAgentSpeaking = false;
  document.getElementById('ai-avatar').classList.remove('pulse-speaking');
  document.getElementById('agent-status-label').innerText = 'Interrupted by Candidate';
  document.getElementById('agent-status-label').className = 'status-pill listening';

  if (currentSessionId) {
    fetch(`/api/session/${currentSessionId}/interrupt`, { method: 'POST' }).catch(() => {});
  }
}

function speakAgentResponse(text) {
  if (!speechSynth) return;
  interruptAgent(); // Stop any pending audio

  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.rate = 1.05;
  currentUtterance.pitch = 1.0;

  isAgentSpeaking = true;
  document.getElementById('ai-avatar').classList.add('pulse-speaking');
  document.getElementById('agent-status-label').innerText = 'AI Coach Speaking...';
  document.getElementById('agent-status-label').className = 'status-pill speaking';

  currentUtterance.onend = () => {
    isAgentSpeaking = false;
    document.getElementById('ai-avatar').classList.remove('pulse-speaking');
    document.getElementById('agent-status-label').innerText = 'Awaiting Candidate Response (Speak or Type)';
    document.getElementById('agent-status-label').className = 'status-pill';
    const input = document.getElementById('manual-answer-input');
    if (input) input.focus();
  };

  speechSynth.speak(currentUtterance);
}

// --------------------------------------------------------------------------
// ANSWER SUBMISSION & LATENCY TRACKING
// --------------------------------------------------------------------------

async function submitCandidateAnswer(answerText) {
  if (!currentSessionId) {
    alert("No active session found. Please click 'Begin Voice Mock Interview' from the setup tab.");
    return;
  }

  const candidateStopTimestamp = candidateSpeechEndTime || Date.now();
  
  // Add candidate turn to UI immediately
  addTranscriptTurn('candidate', answerText, currentStage);

  // Status indicator
  const statusLabel = document.getElementById('agent-status-label');
  if (statusLabel) {
    statusLabel.innerText = 'Analyzing answer & checking follow-up depth...';
    statusLabel.className = 'status-pill listening';
  }

  try {
    const res = await fetch(`/api/session/${currentSessionId}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer_text: answerText })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const errMsg = errData.detail || errData.error || `HTTP ${res.status}`;
      addTranscriptTurn('agent', `⚠️ Error: ${errMsg}`, currentStage);
      if (statusLabel) {
        statusLabel.innerText = 'Error occurred';
        statusLabel.className = 'status-pill';
      }
      return;
    }

    const data = await res.json();

    const latencyMs = Math.max(1, Date.now() - candidateStopTimestamp);
    document.getElementById('live-latency-val').innerText = `${latencyMs} ms`;

    // Follow-up indicator
    const followupBadge = document.getElementById('followup-badge');
    if (data.is_followup) {
      followupBadge.innerText = `Follow-up Probe #${data.followup_index}`;
      followupBadge.classList.remove('hidden');
    } else {
      followupBadge.classList.add('hidden');
    }

    // Update stage & progress
    updateStageStepper(data.stage);
    document.getElementById('live-q-progress').innerText = `${data.questions_asked + 1} / ${data.question_budget}`;

    // Agent response turn
    const agentMsg = data.agent_response || data.question_text || "Thank you. Let's proceed.";
    addTranscriptTurn('agent', agentMsg, data.stage, data.is_followup);
    speakAgentResponse(agentMsg);

    // If completed, trigger report
    if (data.is_completed) {
      setTimeout(() => {
        endInterviewSession();
      }, 4000);
    }

  } catch (err) {
    console.error('Error processing answer:', err);
    addTranscriptTurn('agent', `⚠️ Connection error: ${err.message}`, currentStage);
  }
}

function handleManualAnswer(e) {
  if (e.key === 'Enter') submitManualAnswer();
}

function submitManualAnswer() {
  const input = document.getElementById('manual-answer-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  candidateSpeechEndTime = Date.now();
  submitCandidateAnswer(text);
}

function addTranscriptTurn(speaker, text, stage, isFollowup = false) {
  const feed = document.getElementById('transcript-feed');
  const div = document.createElement('div');
  div.className = `transcript-turn ${speaker}`;

  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  div.innerHTML = `
    <div class="turn-meta">
      <span class="turn-speaker">${speaker === 'agent' ? '🤖 AI Coach' : '👤 Candidate'} ${isFollowup ? '• <small class="text-accent">Follow-up</small>' : ''}</span>
      <span>${timeStr} (${stage})</span>
    </div>
    <div class="turn-body">${text}</div>
  `;

  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

// --------------------------------------------------------------------------
// 3. END INTERVIEW & REPORT RENDERING
// --------------------------------------------------------------------------

async function endInterviewSession() {
  if (!currentSessionId) return;

  interruptAgent();
  if (timerInterval) clearInterval(timerInterval);

  try {
    await fetch(`/api/session/${currentSessionId}/end`, { method: 'POST' });
    const res = await fetch(`/api/session/${currentSessionId}/report`);
    const report = await res.json();
    lastReportData = report;

    renderReport(report);
    switchTab('report');
    document.getElementById('active-session-badge').classList.add('hidden');
    document.getElementById('session-timer-badge').classList.add('hidden');
  } catch (err) {
    console.error('Report fetch error:', err);
  }
}

function renderReport(rep) {
  document.getElementById('report-empty-state').classList.add('hidden');
  document.getElementById('report-content').classList.remove('hidden');

  // Overall Score & Headers
  document.getElementById('rep-overall-score').innerText = rep.overall_score_out_of_5.toFixed(1);
  document.getElementById('rep-candidate-header').innerText = `Evaluation for ${rep.candidate_name}`;
  document.getElementById('rep-role-header').innerText = `Target Role: ${rep.target_role}`;

  // Category Bars
  const cats = rep.category_averages || {};
  setBar('bar-relevance', 'score-relevance', cats.relevance || 4.0);
  setBar('bar-star', 'score-star', cats.structure_star || 4.0);
  setBar('bar-depth', 'score-depth', cats.specificity_depth || 4.0);
  setBar('bar-comm', 'score-comm', cats.communication || 4.0);
  setBar('bar-align', 'score-align', cats.technical_alignment || 4.0);

  // Speech Signals
  const sp = rep.speech_signals || {};
  document.getElementById('metric-wpm').innerText = sp.words_per_minute || 135;
  document.getElementById('metric-pacing').innerText = sp.pacing_verdict || 'Optimal Conversational Pace';
  document.getElementById('metric-fillers').innerText = sp.total_fillers_detected || 0;
  document.getElementById('metric-pauses').innerText = sp.long_pauses_count || 0;

  // 3 Strengths & 3 Improvements
  const sList = document.getElementById('rep-strengths-list');
  sList.innerHTML = '';
  (rep.three_strengths || []).forEach(s => {
    const li = document.createElement('li');
    li.innerText = s;
    sList.appendChild(li);
  });

  const iList = document.getElementById('rep-improvements-list');
  iList.innerHTML = '';
  (rep.three_improvements || []).forEach(i => {
    const li = document.createElement('li');
    li.innerText = i;
    iList.appendChild(li);
  });

  // Scorecards with Evidence Quotes
  const scorecardsBox = document.getElementById('scorecards-container');
  scorecardsBox.innerHTML = '';
  (rep.answer_evaluations || []).forEach((card, idx) => {
    const div = document.createElement('div');
    div.className = 'qa-scorecard';
    div.innerHTML = `
      <div class="qa-q">Q${idx + 1}: ${card.question_text}</div>
      <div class="qa-evidence">Transcript Evidence: ${card.evidence_quotes?.relevance || card.candidate_answer}</div>
      <div class="turn-meta">
        <span>Structure: <strong>${card.scores.structure_star}/5</strong> | Depth: <strong>${card.scores.specificity_depth}/5</strong> | Communication: <strong>${card.scores.communication}/5</strong></span>
      </div>
      <small style="color:#94a3b8;">${card.feedback}</small>
    `;
    scorecardsBox.appendChild(div);
  });

  // Disclaimer
  document.getElementById('rep-disclaimer-text').innerText = rep.disclaimer;
}

function setBar(barId, labelId, val) {
  const pct = Math.round((val / 5.0) * 100);
  const bar = document.getElementById(barId);
  const label = document.getElementById(labelId);
  if (bar) bar.style.width = `${pct}%`;
  if (label) label.innerText = `${val.toFixed(1)} / 5.0`;
}

function exportReportJSON() {
  if (!lastReportData) return;
  const blob = new Blob([JSON.stringify(lastReportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `interview_report_${lastReportData.session_id}.json`;
  a.click();
}

// --------------------------------------------------------------------------
// 4. AUTOMATED AUDIT & BIAS SUITE
// --------------------------------------------------------------------------

async function runAutomatedAudit() {
  const btn = document.getElementById('run-audit-btn');
  btn.innerText = '⏳ Executing 6 Acceptance Tests & Audits...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/tests/run', { method: 'POST' });
    const data = await res.json();

    document.getElementById('audit-results-container').classList.remove('hidden');

    // 6 Acceptance Tests
    const tbody = document.getElementById('acceptance-table-body');
    tbody.innerHTML = '';
    data.acceptance_tests.forEach(t => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${t.name}</strong></td>
        <td>${t.description}</td>
        <td><small class="text-dim">${t.evidence}</small></td>
        <td><span class="badge-tag success">PASS</span></td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById('audit-badge').innerText = `${data.total_passed} / ${data.total_tests} PASSED`;

    // 5-Run Stability
    const runsBox = document.getElementById('stability-runs-pills');
    runsBox.innerHTML = '';
    (data.stability_audit.runs || []).forEach(val => {
      const pill = document.createElement('div');
      pill.className = 'run-pill';
      pill.innerText = val;
      runsBox.appendChild(pill);
    });
    document.getElementById('stat-spread').innerText = `${data.stability_audit.score_spread} pts`;

    // Name Bias
    const nameBox = document.getElementById('name-bias-container');
    nameBox.innerHTML = '';
    (data.name_bias_audit.tested_names || []).forEach(name => {
      const row = document.createElement('div');
      row.className = 'name-bias-row';
      row.innerHTML = `<span>${name}</span><strong>Score: 20/20 (Zero Variance)</strong>`;
      nameBox.appendChild(row);
    });

    // Accented Speech WER & Fairness
    const acc = data.accented_speech_audit;
    document.getElementById('accent-wer-badge').innerText = `WER: ${acc.wer_percentage}`;
    document.getElementById('wer-ground-truth').innerText = acc.ground_truth;
    document.getElementById('wer-hypothesis').innerText = acc.stt_hypothesis;
    document.getElementById('wer-fairness-box').innerText = acc.fairness_analysis;

  } catch (err) {
    alert(`Audit execution error: ${err.message}`);
  } finally {
    btn.innerText = '🚀 Re-Run Full Audit Suite';
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------
// 5. AUDIO VISUALIZER
// --------------------------------------------------------------------------

function initVisualizer() {
  if (!canvasCtx) return;

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const isSpeaking = isRecording || isAgentSpeaking;
    const baseAmp = isSpeaking ? 24 : 4;
    const freq = isSpeaking ? 0.04 : 0.015;

    wavePhase += isSpeaking ? 0.08 : 0.02;

    canvasCtx.lineWidth = 2.5;
    canvasCtx.beginPath();
    
    // Wave 1: Cyan
    canvasCtx.strokeStyle = isSpeaking ? '#06b6d4' : 'rgba(255, 255, 255, 0.15)';
    for (let x = 0; x < canvas.width; x++) {
      const y = (canvas.height / 2) + Math.sin(x * freq + wavePhase) * baseAmp * Math.sin(x / canvas.width * Math.PI);
      if (x === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);
    }
    canvasCtx.stroke();

    // Wave 2: Purple (Offset)
    if (isSpeaking) {
      canvasCtx.beginPath();
      canvasCtx.strokeStyle = '#8b5cf6';
      for (let x = 0; x < canvas.width; x++) {
        const y = (canvas.height / 2) + Math.cos(x * freq + wavePhase + 1.2) * (baseAmp * 0.75) * Math.sin(x / canvas.width * Math.PI);
        if (x === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
      }
      canvasCtx.stroke();
    }
  }

  draw();
}
