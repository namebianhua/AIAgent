/* freeAgent 控制台前端：零依赖、零构建，直接读 SSE 事件流渲染时间线。
 *
 * 设计要点（和文档一致）：
 * 1. 每次工具调用是一张卡片，默认只露一行摘要，点开看完整输入输出；
 * 2. 待确认步骤弹阻塞式窗口（diff 预览 + 倒计时 + 三种粒度），超时自动拒绝；
 * 3. 失败给三个出口：重试 / 跳过 / 改参数后继续；
 * 4. 审计日志里的任意任务可以 2 倍速回放；
 * 5. 状态栏常驻「模型 / 步数 / 耗时 / 最近错误」——用户抱怨"卡住了"时先看这里。
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));

  const S = {
    config: null,
    token: localStorage.getItem('fa.token') || '',
    taskId: null,
    es: null,
    cards: {},
    order: [],
    pending: {},
    pendingAction: null,
    status: 'idle',
    model: '',
    usage: { est_tokens: 0, tool_calls: 0, turns: 0 },
    maxSteps: 15,
    startedAt: 0,
    elapsedMs: 0,
    lastError: '',
    replay: null,
    tick: null,
    openCards: new Set(),
    endedTask: null,
    confirmTimer: null,
  };

  const STATUS_TEXT = {
    idle: '空闲', queued: '排队中', running: '运行中', paused: '已暂停',
    waiting_confirm: '等待确认', waiting_action: '等待处理',
    done: '已完成', failed: '失败', cancelled: '已停止', stopping: '停止中',
  };

  // ------------------------------------------------------------------
  // 基础工具
  // ------------------------------------------------------------------
  function fmtMs(ms) {
    const v = Number(ms) || 0;
    if (v < 1000) return `${Math.round(v)}ms`;
    if (v < 60000) return `${(v / 1000).toFixed(1)}s`;
    const m = Math.floor(v / 60000);
    return `${m}m${Math.round((v % 60000) / 1000)}s`;
  }

  function short(v, n = 60) {
    let text;
    if (typeof v === 'string') text = v;
    else {
      try { text = JSON.stringify(v); } catch (_) { text = String(v); }
    }
    text = String(text ?? '').replace(/\s+/g, ' ');
    return text.length > n ? text.slice(0, n) + '…' : text;
  }

  function argSummary(args) {
    if (!args || typeof args !== 'object') return '';
    return Object.entries(args).map(([k, v]) => `${k}=${short(v, 48)}`).join('  ');
  }

  function withToken(path) {
    if (!S.token) return path;
    return path + (path.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(S.token);
  }

  async function api(path, { method = 'GET', body = null, quiet = false } = {}) {
    const opt = { method, headers: {} };
    if (S.token) opt.headers['X-Auth-Token'] = S.token;
    if (body !== null) {
      opt.headers['Content-Type'] = 'application/json';
      opt.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(withToken(path), opt);
    } catch (e) {
      if (!quiet) toast(`网络请求失败：${e.message}`, 'err');
      throw e;
    }
    if (resp.status === 401) {
      const t = window.prompt('这个控制台启用了访问令牌，请输入 token：', S.token);
      if (t) {
        S.token = t.trim();
        localStorage.setItem('fa.token', S.token);
        return api(path, { method, body, quiet });
      }
    }
    let data = null;
    try { data = await resp.json(); } catch (_) { data = null; }
    if (!resp.ok) {
      const msg = (data && data.error) || `HTTP ${resp.status}`;
      if (!quiet) toast(msg, 'err');
      throw new Error(msg);
    }
    return data;
  }

  // ------------------------------------------------------------------
  // Toast / Modal
  // ------------------------------------------------------------------
  function toast(message, kind = 'info', ms = 3800) {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.innerHTML = esc(message);
    $('toastRoot').appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(12px)';
      el.style.transition = 'all .2s';
      setTimeout(() => el.remove(), 220);
    }, ms);
  }

  let modalCloser = null;

  function openModal(html, { kind = 'generic', closable = true, onClose = null } = {}) {
    closeModal();
    const root = $('modalRoot');
    root.innerHTML = `<div class="overlay" data-kind="${esc(kind)}"><div class="modal">${html}</div></div>`;
    const overlay = root.firstElementChild;
    if (closable) {
      overlay.addEventListener('mousedown', (e) => {
        if (e.target === overlay) closeModal();
      });
    }
    modalCloser = () => {
      root.innerHTML = '';
      modalCloser = null;
      if (onClose) onClose();
    };
    return overlay.querySelector('.modal');
  }

  function closeModal() {
    if (modalCloser) {
      const fn = modalCloser;
      modalCloser = null;
      fn();
    }
  }

  // ------------------------------------------------------------------
  // 视图切换
  // ------------------------------------------------------------------
  function setView(name) {
    const target = (name === 'pending' || name === 'live') ? 'view-run' : `view-${name}`;
    ['view-run', 'view-history', 'view-skills', 'view-settings'].forEach((id) => {
      $(id).hidden = id !== target;
    });
    document.querySelectorAll('.nav-item').forEach((b) => {
      b.classList.toggle('active', b.dataset.view === name);
    });
    if (name === 'history') loadHistory();
    if (name === 'skills') loadSkills();
    if (name === 'settings') loadSettings();
    if (name === 'pending') {
      const first = Object.values(S.pending)[0];
      if (first) showConfirmModal(first);
      else if (S.pendingAction) showActionModal(S.pendingAction);
      else toast('当前没有待确认的操作', 'info');
    }
  }

  // ------------------------------------------------------------------
  // 时间线渲染
  // ------------------------------------------------------------------
  function resetFlow(goalText = '') {
    S.cards = {};
    S.order = [];
    S.openCards.clear();
    $('flow').innerHTML = '';
    $('goalText').textContent = goalText || '还没有任务';
    $('taskChip').textContent = '—';
  }

  function cardKey(ev) {
    if (ev.card) return ev.card;
    if (ev.event === 'tool_start' || ev.event === 'tool_end') {
      return `tool:${ev.plan_step || 0}:${ev.turn || 0}:${ev.tool || ''}`;
    }
    if (ev.event === 'step_start') return `step:${ev.index || 0}`;
    if (ev.event === 'plan') return `plan:${ev.revision || 0}`;
    return `${ev.event}:${ev.seq || Math.random().toString(16).slice(2)}`;
  }

  function upsertCard(ev, replay = false) {
    const id = cardKey(ev);
    let card = S.cards[id];
    if (!card) {
      card = { id, kind: 'tool', status: 'running' };
      S.cards[id] = card;
      S.order.push(id);
      const el = document.createElement('div');
      el.className = 'card';
      el.dataset.card = id;
      el.innerHTML = '<div class="card-head"></div><div class="card-body"></div>';
      $('flow').appendChild(el);
      if (S.order.length === 1) {
        const empty = $('emptyState');
        if (empty) empty.remove();
      }
    }

    switch (ev.event) {
      case 'plan':
        Object.assign(card, {
          kind: 'plan', status: 'done', steps: ev.steps || [], criteria: ev.criteria || '',
          revision: ev.revision || 0, planStep: ev.plan_step || 0,
        });
        break;
      case 'step_start':
        Object.assign(card, { kind: 'step', status: 'running', index: ev.index, total: ev.total, text: ev.step || '', planStep: ev.index || 0 });
        break;
      case 'tool_start':
        Object.assign(card, {
          kind: 'tool', status: 'running', tool: ev.tool || '', args: ev.args || {},
          turn: ev.turn, planStep: ev.plan_step || 0, dangerous: !!ev.dangerous,
          output: '', error: '', elapsed: null,
        });
        break;
      case 'tool_end':
        Object.assign(card, {
          kind: 'tool',
          tool: ev.tool || card.tool || '',
          args: ev.args || card.args || {},
          status: ev.denied ? 'denied' : (ev.cancelled ? 'cancelled' : (ev.skipped ? 'skipped' : (ev.ok ? 'done' : 'failed'))),
          ok: !!ev.ok, output: ev.output || '', error: ev.error || '',
          outputLen: ev.output_len, elapsed: ev.elapsed_ms, attempt: ev.attempt || 1,
          turn: ev.turn ?? card.turn, planStep: ev.plan_step ?? card.planStep,
        });
        break;
      case 'critique':
        Object.assign(card, {
          kind: 'critique', status: ev.ok ? 'done' : 'failed',
          ok: !!ev.ok, reason: ev.reason || '', advice: ev.advice || '',
          planStep: ev.plan_step ?? ev.step ?? 0,
        });
        break;
      case 'replan':
        Object.assign(card, { kind: 'replan', status: 'done', revision: ev.revision, reason: ev.reason || '' });
        break;
      case 'recall':
        Object.assign(card, { kind: 'recall', status: 'done', hits: ev.hits || [] });
        break;
      case 'log':
        Object.assign(card, { kind: 'log', status: 'done', level: ev.level || 'info', message: ev.message || '' });
        break;
      case 'need_confirm':
        card.status = 'pending';
        card.preview = ev.preview;
        break;
      case 'need_action':
        card.status = 'interrupted';
        break;
      default:
        Object.assign(card, { kind: 'log', status: 'done', message: ev.event });
    }

    if (S.openCards.has(id)) card.open = true;
    renderCard(id);

    const flow = $('flow');
    if (!replay && flow.scrollHeight - flow.scrollTop - flow.clientHeight < 160) {
      flow.scrollTop = flow.scrollHeight;
    }
    renderPanel();
  }

  function statusMeta(card) {
    switch (card.status) {
      case 'running': return '执行中…';
      case 'pending': return '等待你确认';
      case 'interrupted': return '等待你处理';
      case 'denied': return '已拒绝';
      case 'cancelled': return '已取消';
      case 'skipped': return '已跳过';
      case 'failed': return card.elapsed != null ? `失败 · ${fmtMs(card.elapsed)}` : '失败';
      case 'done': return card.elapsed != null ? `${fmtMs(card.elapsed)}${card.attempt > 1 ? ` · 第${card.attempt}次` : ''}` : '完成';
      default: return '';
    }
  }

  function closeRunningStep(ok = true) {
    for (let i = S.order.length - 1; i >= 0; i -= 1) {
      const card = S.cards[S.order[i]];
      if (card && card.kind === 'step' && card.status === 'running') {
        card.status = ok ? 'done' : 'failed';
        renderCard(card.id);
        return;
      }
    }
  }

  function renderCard(id) {
    const card = S.cards[id];
    const el = document.querySelector(`[data-card="${id}"]`);
    if (!card || !el) return;
    el.className = `card ${card.status}${card.open ? ' open' : ''}`;
    let title = '';
    let argsText = '';
    let body = '';

    if (card.kind === 'tool') {
      const idx = card.turn ? `<span class="idx">[${card.turn}]</span> ` : '';
      const danger = card.dangerous ? ' <span class="tag high">高危</span>' : '';
      title = `${idx}<span>${esc(card.tool || 'tool')}</span>${danger}`;
      argsText = esc(argSummary(card.args));
      body = `
        <h4>入参</h4>
        <pre class="code">${esc(JSON.stringify(card.args || {}, null, 2))}</pre>
        ${card.output ? `<h4>输出${card.outputLen ? ` · ${card.outputLen} 字符` : ''}</h4><pre class="code ${card.status === 'failed' ? 'err' : 'ok'}">${esc(card.output)}</pre>` : ''}
        ${card.error ? `<h4>错误</h4><pre class="code err">${esc(card.error)}</pre>` : ''}
        ${card.firstError && card.firstError !== card.error ? `<h4>第一次失败</h4><pre class="code err">${esc(card.firstError)}</pre>` : ''}
        ${card.status === 'denied' ? '<h4>说明</h4><pre class="code">用户拒绝了这一步，已把拒绝原因回灌给模型，让它换方案。</pre>' : ''}
        ${card.status === 'skipped' ? '<h4>说明</h4><pre class="code">用户选择跳过，模型会带着「已跳过」的观测继续。</pre>' : ''}
      `;
    } else if (card.kind === 'plan') {
      const rev = card.revision ? ` · 重规划 #${card.revision}` : '';
      title = `<span>计划${esc(rev)}</span>`;
      argsText = `${(card.steps || []).length} 步`;
      body = `
        <h4>步骤</h4>
        <ol class="step-list">${(card.steps || []).map((s, i) => `<li><span class="n">${i + 1}</span><span>${esc(s)}</span></li>`).join('')}</ol>
        ${card.criteria ? `<h4>完成标准</h4><pre class="code">${esc(card.criteria)}</pre>` : ''}
      `;
    } else if (card.kind === 'step') {
      title = `<span>步骤 ${card.index}/${card.total}</span>`;
      argsText = esc(card.text || '');
      body = `<pre class="code">${esc(card.text || '')}</pre>`;
    } else if (card.kind === 'critique') {
      title = `<span>校验${card.ok ? '通过' : '未通过'}</span>`;
      argsText = esc(short(card.reason, 90));
      body = `
        <h4>依据</h4><pre class="code ${card.ok ? 'ok' : 'err'}">${esc(card.reason || '')}</pre>
        ${card.advice ? `<h4>建议</h4><pre class="code">${esc(card.advice)}</pre>` : ''}
      `;
    } else if (card.kind === 'replan') {
      title = `<span>重新规划 #${card.revision}</span>`;
      argsText = esc(short(card.reason, 90));
      body = `<pre class="code">${esc(card.reason || '')}</pre>`;
    } else if (card.kind === 'recall') {
      title = '<span>历史经验</span>';
      argsText = `命中 ${(card.hits || []).length} 条`;
      body = (card.hits || []).length
        ? `<pre class="code">${esc((card.hits || []).map((h) => `${h.score} | ${h.ok ? '成功' : '失败'} | ${h.goal}`).join('\n'))}</pre>`
        : '<pre class="code">没有相似的历史任务</pre>';
    } else {
      title = `<span>${esc(card.level === 'error' ? '错误' : '日志')}</span>`;
      argsText = esc(short(card.message, 120));
      body = `<pre class="code ${card.level === 'error' ? 'err' : ''}">${esc(card.message || '')}</pre>`;
    }

    el.querySelector('.card-head').innerHTML = `
      <span class="dot"></span>
      <span class="card-title">${title}</span>
      <span class="card-args">${argsText}</span>
      <span class="card-meta">${esc(statusMeta(card))}</span>
      <span class="card-chev">▾</span>
    `;
    el.querySelector('.card-body').innerHTML = body;
  }

  function appendAnswerCard(ok, answer) {
    const id = 'answer-' + Date.now();
    const el = document.createElement('div');
    el.className = `card ${ok ? 'done' : 'failed'}`;
    el.dataset.card = id;
    el.innerHTML = `
      <div class="card-head">
        <span class="dot"></span>
        <span class="card-title">${ok ? '最终答复' : '未完成'}</span>
        <span class="card-args">${esc(short(answer, 120))}</span>
        <span class="card-meta">${ok ? '成功' : '失败'}</span>
        <span class="card-chev">▾</span>
      </div>
      <div class="card-body"><pre class="code ${ok ? 'ok' : 'err'}">${esc(answer || '（无内容）')}</pre></div>`;
    el.classList.add('open');
    $('flow').appendChild(el);
  }

  // ------------------------------------------------------------------
  // 事件应用
  // ------------------------------------------------------------------
  function applyEvent(ev, replay = false) {
    switch (ev.event) {
      case 'task_start':
        S.status = 'running';
        S.model = ev.model || S.model;
        S.startedAt = Date.now();
        S.taskId = ev.task_id || S.taskId;
        if (!replay && !(S.taskId === ev.task_id && S.order.length)) {
          resetFlow(ev.goal || '');
          $('taskChip').textContent = ev.task_id || S.taskId || '—';
          $('runBtn').disabled = true;
          $('stopBtn').disabled = false;
          $('pauseBtn').disabled = false;
          $('pauseBtn').textContent = '暂停';
        }
        setStatus('running');
        if (ev.engine === 'demo') $('liveHint').textContent = '演示模型';
        break;

      case 'step_start':
        closeRunningStep();
        upsertCard(ev, replay);
        setStatus(replay ? 'running' : S.status);
        break;

      case 'critique':
        closeRunningStep(ev.ok);
        upsertCard(ev, replay);
        setStatus(replay ? 'running' : S.status);
        break;

      case 'plan':
      case 'tool_start':
      case 'tool_end':
      case 'replan':
      case 'recall':
      case 'log':
        upsertCard(ev, replay);
        setStatus(replay ? 'running' : S.status);
        break;

      case 'need_confirm':
        if (replay) {
          upsertCard({ ...ev, event: 'log', message: `需要确认：${ev.tool}（${short(ev.args, 60)}）`, level: 'warn' }, true);
          break;
        }
        S.pending[ev.cid] = ev;
        setStatus('waiting_confirm');
        updateBadges();
        // 上一个弹窗可能还没关（用户刚点完、服务端已经跑到下一步），此时排队等它关掉
        if (!document.querySelector('.overlay[data-kind="confirm"]')) {
          const first = Object.values(S.pending)[0];
          if (first) showConfirmModal(first);
        }
        break;

      case 'confirm_result':
        if (replay) break;
        delete S.pending[ev.cid];
        if (S.shownCid === ev.cid) {
          clearInterval(confirmCountdown);
          S.shownCid = null;
          closeModal();
          const next = Object.values(S.pending)[0];
          if (next) setTimeout(() => showConfirmModal(next), 150);
          else if (S.pendingAction) setTimeout(() => showActionModal(S.pendingAction), 150);
        }
        updateBadges();
        if (Object.keys(S.pending).length === 0 && !S.pendingAction) setStatus('running');
        toast(
          ev.decision === 'deny'
            ? (ev.timeout ? '已超时，自动拒绝该操作' : '已拒绝该操作')
            : (ev.decision === 'approve_session' ? '已放行（本次任务内同类不再询问）' : '已允许一次'),
          ev.decision === 'deny' ? 'warn' : 'ok',
        );
        break;

      case 'need_action':
        if (replay) {
          upsertCard({ ...ev, event: 'log', message: `等待处理：${ev.tool} 失败 — ${ev.error || ''}`, level: 'error' }, true);
          break;
        }
        S.pendingAction = ev;
        setStatus('waiting_action');
        updateBadges();
        showActionModal(ev);
        break;

      case 'action_result':
        if (replay) break;
        S.pendingAction = null;
        closeModal();
        updateBadges();
        setStatus('running');
        break;

      case 'status':
        if (ev.status === 'paused') setStatus('paused');
        else if (ev.status === 'running') setStatus('running');
        break;

      case 'usage':
        S.usage = { est_tokens: ev.est_tokens || S.usage.est_tokens, turns: ev.turn || S.usage.turns, tool_calls: S.usage.tool_calls };
        if (ev.model) S.model = ev.model;
        renderPanel();
        break;

      case 'snapshot':
        if (ev.view) reconcileSnapshot(ev.view);
        break;

      case 'task_end': {
        if (replay) { appendAnswerCard(!!ev.ok, ev.answer || ''); break; }
        // 服务端发完 task_end 会关流，EventSource 默认会自动重连并重放历史 —— 这里必须去重
        if (S.endedTask === ev.task_id) break;
        S.endedTask = ev.task_id;
        if (S.es) { S.es.close(); S.es = null; }
        const ok = !!ev.ok;
        S.status = ev.stop_reason === 'cancelled' ? 'cancelled' : (ok ? 'done' : 'failed');
        appendAnswerCard(ok, ev.answer || '');
        setStatus(S.status);
        $('runBtn').disabled = false;
        $('stopBtn').disabled = true;
        $('pauseBtn').disabled = true;
        S.pending = {};
        S.pendingAction = null;
        closeModal();
        updateBadges();
        toast(ok ? '任务完成' : `任务结束：${ev.stop_reason || '失败'}`, ok ? 'ok' : 'warn', 5000);
        loadHistoryQuiet();
        break;
      }
      default:
        break;
    }
    updateStats();
  }

  function reconcileSnapshot(view) {
    if (!view) return;
    if (!S.taskId) S.taskId = view.id;
    if (view.timeline && S.order.length === 0) {
      resetFlow(view.goal || '');
      view.timeline.forEach((card) => {
        S.cards[card.id] = { ...card };
        S.order.push(card.id);
        const el = document.createElement('div');
        el.className = `card ${card.status}`;
        el.dataset.card = card.id;
        el.innerHTML = '<div class="card-head"></div><div class="card-body"></div>';
        $('flow').appendChild(el);
        renderCard(card.id);
      });
      const empty = $('emptyState');
      if (empty) empty.remove();
    }
    (view.pending || []).forEach((p) => { S.pending[p.cid] = p; });
    if (view.pending_action) S.pendingAction = view.pending_action;
    S.usage = view.usage || S.usage;
    S.model = view.model || S.model;
    S.status = view.status;
    S.maxSteps = (view.limits && view.limits.max_steps) || S.maxSteps;
    setStatus(view.status);
    renderPanel();
    updateBadges();
  }

  // ------------------------------------------------------------------
  // 确认弹窗（阻塞式）
  // ------------------------------------------------------------------
  function previewHTML(p) {
    if (!p) return '<pre class="code">（无预览）</pre>';
    const kind = p.kind || 'args';
    let body = '';
    if (kind === 'diff') {
      const lines = String(p.body || '').split('\n').map((ln) => {
        let cls = '';
        if (ln.startsWith('+++') || ln.startsWith('---')) cls = 'hunk';
        else if (ln.startsWith('+')) cls = 'add';
        else if (ln.startsWith('-')) cls = 'del';
        else if (ln.startsWith('@@')) cls = 'hunk';
        return `<div class="${cls}">${esc(ln || ' ')}</div>`;
      }).join('');
      body = `<div class="diff">${lines}</div>`;
    } else if (kind === 'command' || kind === 'code' || kind === 'text' || kind === 'files') {
      body = `<pre class="code">${esc(p.body || '')}</pre>`;
    } else {
      body = `<pre class="code">${esc(p.body || '')}</pre>`;
    }
    return `
      <div class="preview-title">
        <b>${esc(p.title || '变更预览')}</b>
        <span class="stat">${esc(p.stat || '')}</span>
      </div>
      ${body}
      ${p.warning ? `<div class="warn-box">⚠ ${esc(p.warning)}</div>` : ''}
    `;
  }

  let confirmCountdown = null;

  function showConfirmModal(p) {
    if (!p) return;
    S.shownCid = p.cid;
    const total = Math.max(5, Number(p.timeout_s) || 30);
    const argsText = JSON.stringify(p.args || {}, null, 2);
    const trust = p.trust_count || 0;
    const dangerTag = p.danger === 'high' ? '<span class="tag high">高风险</span>' : '<span class="tag warn">需确认</span>';

    const modal = openModal(`
      <div class="countdown"><i id="cfBar"></i></div>
      <div class="modal-head">
        <span class="t">这一步要动你的电脑，是否允许？</span>
        ${dangerTag}
      </div>
      <div class="modal-body">
        <div class="preview-title">
          <span class="muted small">操作模式</span>
          <span class="tag">${esc(p.pattern || p.tool || '')}</span>
        </div>
        ${previewHTML(p.preview)}
        <details class="adv">
          <summary>高级：修改参数后执行</summary>
          <textarea class="args-editor" id="cfArgs" spellcheck="false">${esc(argsText)}</textarea>
          <div class="muted small" style="margin-top:6px">改完直接点「允许一次」，会用你改后的参数执行（仍会重新做参数校验）。</div>
        </details>
        <div class="muted small" style="margin-top:10px">
          ${trust > 0 ? `同类操作（${esc(p.pattern || '')}）已确认 ${trust} 次，再确认 ${Math.max(0, 3 - trust)} 次将自动放行。` : '同一个操作确认 3 次后会自动升为白名单，可在设置里清空。'}
        </div>
      </div>
      <div class="modal-foot">
        <span class="note" id="cfCountdown">${total}s 后自动拒绝</span>
        <button class="btn danger" id="cfDeny">拒绝</button>
        <button class="btn ghost" id="cfOnce">允许一次</button>
        <button class="btn primary" id="cfSession">本任务内允许同类</button>
      </div>
    `, { kind: 'confirm', closable: false });

    const deadline = Date.now() + total * 1000;
    clearInterval(confirmCountdown);
    confirmCountdown = setInterval(() => {
      const left = Math.max(0, deadline - Date.now());
      const bar = $('cfBar');
      if (!bar) { clearInterval(confirmCountdown); return; }
      bar.style.width = `${(left / (total * 1000)) * 100}%`;
      const label = $('cfCountdown');
      if (label) label.textContent = `${Math.ceil(left / 1000)}s 后自动拒绝`;
      if (left <= 0) {
        clearInterval(confirmCountdown);
      }
    }, 200);

    const collect = () => {
      const ta = $('cfArgs');
      if (!ta) return null;
      const raw = ta.value.trim();
      if (raw === argsText.trim()) return null;
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
        toast('参数必须是 JSON 对象', 'err');
      } catch (e) {
        toast('参数不是合法 JSON：' + e.message, 'err');
      }
      return undefined;
    };

    const send = (decision, remember) => {
      const args = collect();
      if (args === undefined) return;
      clearInterval(confirmCountdown);
      api('/api/confirm', {
        method: 'POST',
        body: { task_id: S.taskId, cid: p.cid, decision, remember, args },
      }).then(() => {
        delete S.pending[p.cid];
        if (S.shownCid === p.cid) {
          clearInterval(confirmCountdown);
          S.shownCid = null;
          closeModal();
        }
        updateBadges();
      }).catch(() => {});
    };

    modal.querySelector('#cfDeny').onclick = () => send('deny', false);
    modal.querySelector('#cfOnce').onclick = () => send('approve', false);
    modal.querySelector('#cfSession').onclick = () => send('approve_session', true);
  }

  // ------------------------------------------------------------------
  // 失败干预
  // ------------------------------------------------------------------
  function showActionModal(p) {
    if (!p) return;
    const argsText = JSON.stringify(p.args || {}, null, 2);
    const modal = openModal(`
      <div class="modal-head">
        <span class="t">这一步失败了，你想怎么走？</span>
        <span class="tag high">${esc(p.tool || '')}</span>
      </div>
      <div class="modal-body">
        <h4 style="font-size:11px;color:var(--muted);margin-bottom:6px">错误</h4>
        <pre class="code err">${esc(p.error || '（无错误信息）')}</pre>
        <h4 style="font-size:11px;color:var(--muted);margin:10px 0 6px">当前参数</h4>
        <pre class="code">${esc(argsText)}</pre>
        <details class="adv" open>
          <summary>改参数后继续（最常用：只差一个路径）</summary>
          <textarea class="args-editor" id="acArgs" spellcheck="false">${esc(argsText)}</textarea>
        </details>
        <div class="muted small" style="margin-top:10px">重试 = 原参数再跑一次；跳过 = 记一条「已跳过」让模型继续；改参数 = 用你改的 JSON 重跑这一步。</div>
      </div>
      <div class="modal-foot">
        <span class="note">最多等 5 分钟，超时按继续处理</span>
        <button class="btn ghost" id="acAbort">放弃任务</button>
        <button class="btn ghost" id="acSkip">跳过这一步</button>
        <button class="btn ghost" id="acRetry">重试</button>
        <button class="btn primary" id="acFix">改参数后继续</button>
      </div>
    `, { kind: 'action', closable: false });

    const send = (action) => {
      let args = null;
      if (action === 'fix') {
        const raw = ($('acArgs') || {}).value || '';
        try {
          args = JSON.parse(raw);
          if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('需要 JSON 对象');
        } catch (e) {
          toast('参数不是合法 JSON：' + e.message, 'err');
          return;
        }
      }
      api('/api/action', { method: 'POST', body: { task_id: S.taskId, cid: p.cid, action, args } })
        .then(() => { S.pendingAction = null; closeModal(); updateBadges(); setStatus('running'); })
        .catch(() => {});
    };

    modal.querySelector('#acRetry').onclick = () => send('retry');
    modal.querySelector('#acSkip').onclick = () => send('skip');
    modal.querySelector('#acFix').onclick = () => send('fix');
    modal.querySelector('#acAbort').onclick = () => send('abort');
  }

  // ------------------------------------------------------------------
  // 状态 / 面板
  // ------------------------------------------------------------------
  function setStatus(status) {
    S.status = status;
    const text = STATUS_TEXT[status] || status;
    $('stStatus').textContent = text;
    const dot = $('connDot');
    dot.className = 'status-dot ' + (
      status === 'running' ? 'busy'
        : status === 'waiting_confirm' || status === 'waiting_action' || status === 'paused' ? 'warn'
        : status === 'done' ? 'live' : ''
    );
    $('pStatus').textContent = text;
    if (status === 'done' || status === 'failed' || status === 'cancelled') {
      $('pauseBtn').disabled = true;
      $('stopBtn').disabled = true;
      $('runBtn').disabled = false;
      stopTick();
    } else if (status !== 'idle') {
      startTick();
    }
  }

  function updateStats() {
    const toolCards = Object.values(S.cards).filter((c) => c.kind === 'tool');
    const steps = toolCards.length;
    const maxSteps = S.maxSteps || 15;
    $('stSteps').textContent = `${steps} / ${maxSteps}`;
    $('pSteps').textContent = `${steps} / ${maxSteps}`;
    $('stepMeter').style.width = `${Math.min(100, (steps / maxSteps) * 100)}%`;
    $('stElapsed').textContent = fmtMs(S.elapsedMs);
    $('pElapsed').textContent = fmtMs(S.elapsedMs);
    $('pModel').textContent = S.model || '—';
    $('stModel').textContent = S.model || '—';
    $('pTokens').textContent = S.usage.est_tokens ? `≈${S.usage.est_tokens}` : '—';
    const artifacts = [...new Set(Object.values(S.cards)
      .filter((c) => c.kind === 'tool' && c.tool === 'write_file' && c.ok)
      .map((c) => (c.args || {}).path).filter(Boolean))];
    $('pArtifacts').textContent = artifacts.length ? artifacts.join(', ') : '—';
    $('stError').textContent = S.lastError ? short(S.lastError, 46) : '—';
    $('stError').style.color = S.lastError ? 'var(--red)' : '';
    updateBadges();
  }

  function startTick() {
    if (S.tick) return;
    S.tick = setInterval(() => {
      if (S.status === 'running' || S.status === 'waiting_confirm' || S.status === 'waiting_action' || S.status === 'paused') {
        if (S.startedAt) S.elapsedMs = Date.now() - S.startedAt;
      }
      $('stElapsed').textContent = fmtMs(S.elapsedMs);
      $('pElapsed').textContent = fmtMs(S.elapsedMs);
    }, 500);
  }

  function stopTick() {
    clearInterval(S.tick);
    S.tick = null;
  }

  function updateBadges() {
    const pending = Object.keys(S.pending).length + (S.pendingAction ? 1 : 0);
    const badge = $('badgeConfirm');
    badge.textContent = pending;
    badge.classList.toggle('hidden', pending === 0);
    const running = (S.status === 'running' || S.status === 'waiting_confirm' || S.status === 'waiting_action') ? 1 : 0;
    const rb = $('badgeRunning');
    rb.textContent = running;
    rb.classList.toggle('hidden', running === 0);
    renderBanner(pending);
  }

  function renderBanner(pending) {
    const area = $('bannerArea');
    if (!pending) { area.innerHTML = ''; return; }
    const first = Object.values(S.pending)[0];
    const label = first ? `「${first.tool}」等待你确认` : '失败步骤等待你处理';
    area.innerHTML = `
      <div class="banner">
        <b>你有 ${pending} 个待处理</b>
        <span class="muted">${esc(label)}</span>
        <span class="spacer"></span>
        <button class="btn small primary" id="bannerGo">去处理</button>
      </div>`;
    const go = $('bannerGo');
    if (go) go.onclick = () => {
      if (Object.keys(S.pending).length) showConfirmModal(Object.values(S.pending)[0]);
      else if (S.pendingAction) showActionModal(S.pendingAction);
    };
  }

  function renderPanel() {
    updateStats();
    if (!S.config) return;
    $('wsPath').textContent = '工作区：' + (S.config.workspace || '—');
    $('wsPath').title = S.config.workspace || '';
    $('modeLabel').textContent = S.config.mode;
    const models = (S.config.models || []).map((m) => `
      <div class="row"><span class="k">${esc(m.label || m.name)}</span>
      <span class="v">${m.available ? (m.primary ? '主' : '备') : '未配置'}</span></div>`).join('');
    if (models) $('modelList').innerHTML = models;
  }

  // ------------------------------------------------------------------
  // 右侧：文件树 / 快照 / 信任
  // ------------------------------------------------------------------
  async function loadTree(path = '.') {
    try {
      const d = await api('/api/files?path=' + encodeURIComponent(path) + '&depth=2', { quiet: true });
      $('fileTree').innerHTML = renderTree(d.tree.children || []);
    } catch (_) {
      $('fileTree').innerHTML = '<div class="muted small">读取失败</div>';
    }
  }

  function renderTree(nodes) {
    if (!nodes.length) return '<div class="muted small">（空）</div>';
    return nodes.map((n) => {
      if (n.type === 'dir') {
        return `<div class="node dir" data-dir="${esc(n.path)}">▸ ${esc(n.name)}/</div>
          <div class="children hidden" data-children="${esc(n.path)}">${renderTree(n.children || [])}</div>`;
      }
      if (n.type === 'more') return '<div class="node muted">…</div>';
      const size = n.size > 1024 ? `${(n.size / 1024).toFixed(1)}K` : `${n.size}B`;
      return `<div class="node file" data-file="${esc(n.path)}">· ${esc(n.name)} <span class="muted">${size}</span></div>`;
    }).join('');
  }

  async function loadSnapshots() {
    try {
      const d = await api('/api/snapshots', { quiet: true });
      const rows = d.snapshots || [];
      $('snapList').innerHTML = rows.length
        ? rows.map((s) => `<div class="row"><span class="k mono small">${esc(s.name)}</span>
            <span class="v"><button class="link" data-rollback="${esc(s.name)}">回滚</button></span></div>`).join('')
        : '<div class="muted small">暂无快照（运行前勾选「任务前打快照」）</div>';
    } catch (_) {
      $('snapList').innerHTML = '<div class="muted small">读取失败</div>';
    }
  }

  async function loadTrust() {
    try {
      const d = await api('/api/trust', { quiet: true });
      const rules = d.rules || {};
      const keys = Object.keys(rules);
      if (!keys.length) {
        $('trustList').innerHTML = '<div class="muted small">确认 3 次后同类操作自动放行</div>';
        return;
      }
      $('trustList').innerHTML = keys.map((k) => {
        const r = rules[k] || {};
        return `<div class="row"><span class="k mono small">${esc(k)}</span>
          <span class="v">${r.trusted ? '<span class="trust-pill">已信任</span>' : `已确认 ${r.count || 0}/3`}</span></div>`;
      }).join('');
    } catch (_) {}
  }

  // ------------------------------------------------------------------
  // 运行 / 停止 / 暂停
  // ------------------------------------------------------------------
  async function runTask() {
    const goal = $('goal').value.trim();
    if (!goal) { toast('请先输入目标', 'warn'); $('goal').focus(); return; }

    const options = {
      autonomous: $('optAutonomous').checked,
      snapshot: $('optSnapshot').checked,
      pause_on_failure: $('optPauseFail').checked,
      confirm_writes: $('optConfirmWrites').checked,
      readonly: $('optReadonly').checked,
      auto_yes: $('optAutoYes').checked,
      max_steps: Number($('optMaxSteps').value) || 15,
    };
    S.maxSteps = options.max_steps;
    S.lastError = '';
    S.pending = {};
    S.pendingAction = null;
    S.usage = { est_tokens: 0, tool_calls: 0, turns: 0 };
    S.elapsedMs = 0;
    S.endedTask = null;
    resetFlow(goal);
    S.taskId = null;
    setStatus('running');

    try {
      const d = await api('/api/run', { method: 'POST', body: { goal, options } });
      S.taskId = d.task_id;
      $('taskChip').textContent = d.task_id;
      connect(d.task_id);
    } catch (_) {
      setStatus('idle');
      $('runBtn').disabled = false;
    }
  }

  function connect(taskId) {
    if (S.es) { S.es.close(); S.es = null; }
    const es = new EventSource(withToken('/api/stream?id=' + encodeURIComponent(taskId)));
    S.es = es;
    es.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch (_) { return; }
      applyEvent(ev);
    };
    es.onerror = () => {
      // 任务结束时服务端会关流，这是正常现象；别让它自动重连把历史再放一遍
      if (S.endedTask === taskId) { es.close(); return; }
      if (S.status === 'done' || S.status === 'failed' || S.status === 'cancelled') return;
      $('connDot').className = 'status-dot';
      $('liveHint').textContent = '连接中断，正在重连…';
    };
    es.onopen = () => { $('liveHint').textContent = ''; };
  }

  async function stopTask() {
    if (!S.taskId) return;
    await api('/api/stop?id=' + encodeURIComponent(S.taskId), { method: 'POST', body: {} });
    toast('已请求停止：当前工具调用结束后收尾', 'warn');
    setStatus('stopping');
  }

  async function pauseTask() {
    if (!S.taskId) return;
    if (S.status === 'paused') {
      await api('/api/resume?id=' + encodeURIComponent(S.taskId), { method: 'POST', body: {} });
      $('pauseBtn').textContent = '暂停';
      setStatus('running');
    } else {
      await api('/api/pause?id=' + encodeURIComponent(S.taskId), { method: 'POST', body: {} });
      $('pauseBtn').textContent = '继续';
      setStatus('paused');
    }
  }

  // ------------------------------------------------------------------
  // 审计日志 + 回放
  // ------------------------------------------------------------------
  async function loadHistory() {
    try {
      const d = await api('/api/traces?limit=40');
      const rows = d.traces || [];
      $('traceBody').innerHTML = rows.length ? rows.map((r) => {
        const ok = r.ok === true;
        const result = ok ? '<span class="tag ok">成功</span>'
          : `<span class="tag high">${esc(r.stop_reason || '未完成')}</span>`;
        return `<tr class="clickable" data-trace="${esc(r.task_id)}">
          <td class="mono small">${esc(r.task_id)}</td>
          <td class="goal">${esc(r.goal || '—')}</td>
          <td>${result}</td>
          <td class="mono small">${r.tools || 0}</td>
          <td class="muted small">${esc(r.mtime || '')}</td>
        </tr>`;
      }).join('') : '<tr><td colspan="5" class="muted">还没有轨迹（跑一个任务就有了）</td></tr>';
    } catch (_) {
      $('traceBody').innerHTML = '<tr><td colspan="5" class="muted">读取失败</td></tr>';
    }
  }

  function loadHistoryQuiet() {
    if (!$('view-history').hidden) loadHistory();
  }

  async function openTrace(taskId) {
    try {
      const d = await api('/api/trace?id=' + encodeURIComponent(taskId));
      setView('run');
      startReplay(d.events || [], taskId);
    } catch (_) {}
  }

  function startReplay(events, taskId) {
    if (!events.length) { toast('轨迹为空', 'warn'); return; }
    if (S.es) { S.es.close(); S.es = null; }
    S.replay = { events, i: 0, speed: 2, playing: true, timer: null, taskId };
    S.pending = {};
    S.pendingAction = null;
    S.cards = {};
    S.order = [];
    S.openCards.clear();
    $('flow').innerHTML = '';
    $('replayBar').classList.remove('hidden');
    $('replayPlay').textContent = '暂停';
    $('replaySpeed').textContent = '2×';
    $('replayProgress').style.width = '0%';
    $('replayLabel').textContent = '0%';
    $('taskChip').textContent = taskId || '—';

    const start = events.find((e) => e.event === 'task_start');
    if (start) $('goalText').textContent = start.goal || '（历史任务）';
    S.startedAt = 0;
    S.status = 'running';
    setStatus('running');
    stepReplay();
  }

  function stepReplay() {
    const r = S.replay;
    if (!r || !r.playing) return;
    if (r.i >= r.events.length) {
      r.playing = false;
      $('replayPlay').textContent = '重播';
      $('replayLabel').textContent = '100%';
      setStatus('done');
      return;
    }
    const ev = r.events[r.i];
    const prev = r.i > 0 ? (r.events[r.i - 1].t_ms || 0) : 0;
    const gap = Math.max(60, Math.min(1400, ((ev.t_ms || 0) - prev) / r.speed));
    r.timer = setTimeout(() => {
      if (!S.replay || !S.replay.playing) return;
      applyEvent(ev, true);
      r.i += 1;
      const pct = Math.round((r.i / r.events.length) * 100);
      $('replayProgress').style.width = pct + '%';
      $('replayLabel').textContent = pct + '%';
      if (ev.event === 'task_end') {
        r.playing = false;
        $('replayPlay').textContent = '重播';
        $('replayLabel').textContent = '100%';
        setStatus('done');
        stopTick();
      } else {
        stepReplay();
      }
    }, gap);
  }

  function toggleReplay() {
    const r = S.replay;
    if (!r) return;
    if (r.playing) {
      r.playing = false;
      clearTimeout(r.timer);
      $('replayPlay').textContent = '继续';
      return;
    }
    if (r.i >= r.events.length) { startReplay(r.events, r.taskId); return; }
    r.playing = true;
    $('replayPlay').textContent = '暂停';
    stepReplay();
  }

  function exitReplay() {
    if (!S.replay) return;
    clearTimeout(S.replay.timer);
    S.replay = null;
    $('replayBar').classList.add('hidden');
    resetFlow();
    setStatus('idle');
    $('runBtn').disabled = false;
  }

  // ------------------------------------------------------------------
  // 技能 / 设置
  // ------------------------------------------------------------------
  async function loadSkills() {
    try {
      const d = await api('/api/skills');
      const skills = d.skills || [];
      $('skillGrid').innerHTML = skills.length ? skills.map((s) => `
        <div class="tile">
          <h4>skill_${esc(s.name)}</h4>
          <p>${esc(s.description || '')}</p>
          <div class="flow-line">${esc((s.steps || []).join(' → '))}</div>
          <div class="flow-line">参数：${esc((s.params || []).join(', ') || '—')}</div>
          <div class="flow-line">调用 ${s.uses} 次 / 成功 ${s.successes} 次</div>
          <div style="margin-top:10px"><button class="btn small ghost" data-del-skill="${esc(s.name)}">删除</button></div>
        </div>`).join('')
        : '<div class="empty" style="margin-top:20px">技能库为空。用 <code>--save-skill NAME</code> 跑一个成功任务即可固化。</div>';
    } catch (_) {
      $('skillGrid').innerHTML = '<div class="muted">读取失败</div>';
    }
  }

  async function loadSettings() {
    if (!S.config) return;
    const c = S.config;
    const localOn = c.local && c.local.enabled;
    $('settingsBody').innerHTML = `
      <div class="settings-block">
        <h3>运行环境</h3>
        <div class="row"><span>版本</span><span class="v">${esc(c.version)}</span></div>
        <div class="row"><span>工作区</span><span class="v">${esc(c.workspace)}</span></div>
        <div class="row"><span>模型模式</span><span class="v">${esc(c.mode)}</span></div>
        <div class="row"><span>确认超时</span><span class="v">${c.confirm.timeout}s</span></div>
        <div class="row"><span>写文件也确认</span><span class="v">${c.confirm.confirm_writes ? '是' : '否'}</span></div>
        <div class="row"><span>访问令牌</span><span class="v">${c.token_required ? '已启用' : '未启用（仅本机）'}</span></div>
        <div class="row"><span>演示模式</span><span class="v">${c.features.demo ? '是' : '否'}</span></div>
      </div>
      ${localOn ? `
      <div class="settings-block">
        <h3>模型路由（本地优先）</h3>
        <div class="row"><span>本地模型</span><span class="v" id="mLocal">${esc(c.local.model || '—')}</span></div>
        <div class="row"><span>本地状态</span><span class="v" id="mStatus">探测中…</span></div>
        <div class="row"><span>路由偏好</span><span class="v">
          <button class="btn small ghost" id="preferBtn">—</button></span></div>
        <div class="row"><span>用量统计</span><span class="v small" id="mStats">—</span></div>
        <div class="sub small">本地 Ollama 优先；超时 / 连不上 / 5xx 自动降级到后备通道，4xx 不降级（改了也是再错一次）。连续解析失败会临时切大模型重算。</div>
      </div>` : ''}
      <div class="settings-block">
        <h3>电脑操控</h3>
        <div class="row"><span>当前作用域</span><span class="v" id="pcScope">${c.computer && c.computer.enabled
          ? '<span class="tag high">整台电脑</span>' : '<span class="tag ok">仅工作区</span>'}</span></div>
        <div class="row"><span>作用域切换</span><span class="v">
          <button class="btn small ${c.computer && c.computer.enabled ? 'danger' : 'ghost'}" id="pcToggle"
            ${c.computer && !c.computer.supported ? 'disabled title="当前平台不支持"' : ''}>
            ${c.computer && c.computer.enabled ? '收回工作区沙箱' : '放开到整台电脑'}</button></span></div>
        <div class="sub small">放开后 agent 可访问工作区外的文件（绝对路径），并启用 sys_*（系统查询）与 ui_*（鼠标/键盘/窗口/程序等真实桌面动作）工具。安全线：Windows / Program Files / ProgramData / 盘符根目录禁写；工作区外禁止递归删除；所有 ui_* 动作与工作区外写入都会弹人工确认。切换只对之后提交的新任务生效。</div>
      </div>
      <div class="settings-block">
        <h3>上限</h3>
        <div class="row"><span>单任务最大步数</span><span class="v">${c.limits.max_steps}</span></div>
        <div class="row"><span>计划最多几步</span><span class="v">${c.limits.max_plan_steps}</span></div>
        <div class="row"><span>最多重规划</span><span class="v">${c.limits.max_replans}</span></div>
        <div class="row"><span>时间预算</span><span class="v">${c.limits.max_seconds}s</span></div>
      </div>
      <div class="settings-block">
        <h3>确认策略</h3>
        <div class="sub small">规则自上而下匹配，首条命中生效：deny 静默拒绝、allow 静默放行、ask 弹人工确认（批准 ${c.confirm.trust_threshold} 次自动升为 allow）。工具与参数都支持 glob（如 <code>read_*</code>、<code>{"cmd":"pip install *"}</code>）。</div>
        <table class="table" style="margin-top:10px">
          <thead><tr><th style="width:70px">顺序</th><th>工具</th><th>参数模式</th><th>动作</th><th>命中</th><th>范围</th><th></th></tr></thead>
          <tbody id="policyBody"><tr><td colspan="7" class="muted">加载中…</td></tr></tbody>
        </table>
        <div class="row" style="margin-top:10px;flex-wrap:wrap">
          <input id="pTool" placeholder="工具 glob，如 run_shell" style="width:150px">
          <input id="pParam" placeholder='参数模式 JSON，如 {"cmd":"pip install *"}' style="width:230px">
          <select id="pAction"><option value="ask">ask 询问</option><option value="allow">allow 放行</option><option value="deny">deny 拒绝</option></select>
          <button class="btn small ghost" id="pAdd">新增规则</button>
          <button class="btn small ghost" id="pExport">导出</button>
          <button class="btn small ghost" id="pImport">导入</button>
        </div>
        <div class="row" style="margin-top:8px;flex-wrap:wrap">
          <span class="small muted" style="margin-right:6px">模拟匹配：</span>
          <input id="simTool" placeholder="工具名，如 run_shell" style="width:140px">
          <input id="simArgs" placeholder='参数 JSON，如 {"cmd":"pip install numpy"}' style="width:230px">
          <button class="btn small ghost" id="pSim">测试</button>
          <span id="simResult" class="small" style="flex:1"></span>
        </div>
      </div>
      <div class="settings-block">
        <h3>安全</h3>
        <div class="row"><span>旧版计数信任</span><span class="v">确认 3 次同类自动放行（无规则命中时兜底）</span></div>
        <div class="row"><span>清空全部信任数据</span><span class="v"><button class="btn small danger" id="clearTrust">清空</button></span></div>
        <div class="row"><span>立即创建快照</span><span class="v"><button class="btn small ghost" id="makeSnap">打快照</button></span></div>
      </div>`;

    const ct = $('clearTrust');
    if (ct) ct.onclick = () => api('/api/trust', { method: 'DELETE' }).then(() => { toast('已清空（恢复默认策略）', 'ok'); loadTrust(); loadPolicies(); });
    const ms = $('makeSnap');
    if (ms) ms.onclick = () => api('/api/snapshot', { method: 'POST', body: {} })
      .then((d) => { toast(d.ok ? `已创建 ${d.snapshot}` : '快照失败', d.ok ? 'ok' : 'err'); loadSnapshots(); });

    if (localOn) {
      const pb = $('preferBtn');
      if (pb) pb.onclick = async () => {
        try {
          const st = await api('/api/model/status');
          const d = await api('/api/model/prefer', { method: 'POST', body: { prefer_local: !st.prefer_local } });
          toast(d.prefer_local ? '已切回本地优先' : '已切为后备通道优先', 'ok');
          loadModelStatus();
        } catch (_) {}
      };
      loadModelStatus();
    }
    const pcBtn = $('pcToggle');
    if (pcBtn) pcBtn.onclick = async () => {
      try {
        const st = await api('/api/scope', { quiet: true });
        const d = await api('/api/scope', { method: 'POST', body: { enabled: !st.enabled } });
        toast(d.enabled ? '已放开到整台电脑（新任务生效）' : '已收回工作区沙箱（新任务生效）', 'ok');
        // 刷新缓存的配置（loadSettings 渲染依赖 S.config），再重绘设置页
        S.config = await api('/api/config', { quiet: true });
        loadSettings();
      } catch (_) { toast('切换失败', 'err'); }
    };
    loadPolicies();
    bindPolicyEditor();
  }

  // ------------------------------------------------------------------
  // 模型路由状态 / 确认策略规则表
  // ------------------------------------------------------------------
  async function loadModelStatus() {
    if (!$('mStatus')) return;
    try {
      const st = await api('/api/model/status', { quiet: true });
      const ok = st.local && st.local.available;
      $('mStatus').innerHTML = ok
        ? '<span class="tag ok">在线</span>'
        : '<span class="tag warn">离线（走后备通道）</span>';
      const pb = $('preferBtn');
      if (pb) pb.textContent = st.prefer_local ? '本地优先（点击切换）' : '后备优先（点击切换）';
      const s = st.stats || {};
      $('mStats').textContent =
        `本地 ${s.local_calls || 0} · API ${s.api_calls || 0} · 网页 ${s.web_calls || 0}`
        + ` · 降级 ${s.fallbacks || 0} · 解析失败 ${s.parse_errors || 0}`
        + (st.last_latency_ms ? ` · 上次推理 ${(st.last_latency_ms / 1000).toFixed(1)}s` : '');
    } catch (_) {
      $('mStatus').textContent = '探测失败';
    }
  }

  async function loadPolicies() {
    const body = $('policyBody');
    if (!body) return;
    try {
      const d = await api('/api/trust/policies', { quiet: true });
      const rules = d.policies || [];
      if (!rules.length) {
        body.innerHTML = '<tr><td colspan="7" class="muted">（空表）</td></tr>';
        return;
      }
      body.innerHTML = rules.map((r, i) => {
        const cls = r.action === 'allow' ? 'ok' : (r.action === 'deny' ? 'high' : 'warn');
        const scope = r.scope === 'session' ? '会话' : '全局';
        const expired = r.expires_at && (Date.now() / 1000 > r.expires_at);
        return `<tr>
          <td class="mono">${i + 1}
            <button class="link small" data-up="${esc(r.id)}" title="上移">↑</button>
            <button class="link small" data-down="${esc(r.id)}" title="下移">↓</button></td>
          <td class="mono">${esc(r.tool)}</td>
          <td class="mono small">${esc(Object.keys(r.param_pattern || {}).length
            ? JSON.stringify(r.param_pattern) : '（任意参数）')}</td>
          <td><span class="tag ${cls}">${r.action}</span>${r.promoted ? ' <span class="small muted" title="由 ask 自动晋升">↑自动</span>' : ''}</td>
          <td class="mono">${r.hits}</td>
          <td class="small">${expired ? '<span class="tag high">已过期</span>' : scope}</td>
          <td>
            <button class="link small" data-toggle="${esc(r.id)}">${r.enabled ? '停用' : '启用'}</button>
            <button class="link small" data-del="${esc(r.id)}">删除</button>
          </td>
        </tr>`;
      }).join('');
    } catch (_) {
      body.innerHTML = '<tr><td colspan="7" class="muted">读取失败</td></tr>';
    }
  }

  function bindPolicyEditor() {
    const body = $('policyBody');
    if (!body) return;
    body.onclick = async (e) => {
      const up = e.target.closest('[data-up]');
      const down = e.target.closest('[data-down]');
      const del = e.target.closest('[data-del]');
      const tog = e.target.closest('[data-toggle]');
      if (up || down) {
        const rows = [...body.querySelectorAll('tr')];
        const ids = rows.map((tr) => (tr.querySelector('[data-up]') || {}).dataset && tr.querySelector('[data-up]').dataset.up).filter(Boolean);
        const id = (up || down).dataset.up || (up || down).dataset.down;
        const i = ids.indexOf(id);
        const j = (up ? Math.max(0, i - 1) : Math.min(ids.length - 1, i + 1));
        if (i !== j) { [ids[i], ids[j]] = [ids[j], ids[i]]; }
        try { await api('/api/trust/policies/reorder', { method: 'POST', body: { ids } }); loadPolicies(); } catch (_) {}
        return;
      }
      if (del) {
        try {
          const d = await api('/api/trust/policies?id=' + encodeURIComponent(del.dataset.del), { method: 'DELETE' });
          toast(d.ok ? '已删除' : '删除失败', d.ok ? 'ok' : 'err');
          loadPolicies();
        } catch (_) {}
        return;
      }
      if (tog) {
        try {
          const d = await api('/api/trust/policies');
          const rule = (d.policies || []).find((r) => r.id === tog.dataset.toggle);
          if (rule) {
            await api('/api/trust/policies/update', { method: 'POST', body: { id: rule.id, enabled: !rule.enabled } });
            loadPolicies();
          }
        } catch (_) {}
      }
    };

    const add = $('pAdd');
    if (add) add.onclick = async () => {
      const tool = ($('pTool').value || '').trim();
      if (!tool) { toast('工具 glob 不能为空', 'warn'); return; }
      let pp = {};
      const raw = ($('pParam').value || '').trim();
      if (raw) {
        try { pp = JSON.parse(raw); } catch (_) { toast('参数模式不是合法 JSON', 'err'); return; }
      }
      try {
        await api('/api/trust/policies', {
          method: 'POST',
          body: { tool, param_pattern: pp, action: $('pAction').value },
        });
        toast('已新增（插到最前，优先级最高）', 'ok');
        $('pTool').value = ''; $('pParam').value = '';
        loadPolicies();
      } catch (_) {}
    };

    const sim = $('pSim');
    if (sim) sim.onclick = async () => {
      const tool = ($('simTool').value || '').trim();
      if (!tool) { toast('请填工具名', 'warn'); return; }
      let args = {};
      try { args = JSON.parse(($('simArgs').value || '').trim() || '{}'); } catch (_) {
        $('simResult').innerHTML = '<span class="tag high">参数 JSON 不合法</span>';
        return;
      }
      try {
        const d = await api('/api/trust/policies/dry-run', { method: 'POST', body: { tool, args } });
        $('simResult').innerHTML = d.matched
          ? `命中 <code>${esc(d.rule.id)}</code> → <span class="tag ${d.action === 'allow' ? 'ok' : (d.action === 'deny' ? 'high' : 'warn')}">${esc(d.action)}</span>`
          : '无匹配 → <span class="tag warn">ask（默认人工确认）</span>';
      } catch (_) {}
    };

    const exp = $('pExport');
    if (exp) exp.onclick = async () => {
      try {
        const d = await api('/api/trust/policies/export', { quiet: true });
        const blob = new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'trust-rules.json';
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (_) {}
    };

    const imp = $('pImport');
    if (imp) imp.onclick = () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json';
      input.onchange = async () => {
        if (!input.files.length) return;
        try {
          const payload = JSON.parse(await input.files[0].text());
          const d = await api('/api/trust/policies/import', { method: 'POST', body: { payload } });
          toast(d.ok ? `已导入 ${d.count} 条规则` : '导入失败（文件格式不对）', d.ok ? 'ok' : 'err');
          loadPolicies();
        } catch (err) {
          toast('导入失败：' + err, 'err');
        }
      };
      input.click();
    };
  }

  // ------------------------------------------------------------------
  // 事件绑定
  // ------------------------------------------------------------------
  function bind() {
    $('runBtn').onclick = runTask;
    $('stopBtn').onclick = stopTask;
    $('pauseBtn').onclick = pauseTask;
    $('optBtn').onclick = () => $('optPanel').classList.toggle('hidden');
    $('refreshTree').onclick = () => loadTree('.');
    $('snapNow').onclick = () => api('/api/snapshot', { method: 'POST', body: {} })
      .then((d) => { toast(d.ok ? `已创建 ${d.snapshot}` : '快照失败', d.ok ? 'ok' : 'err'); loadSnapshots(); });

    $('goal').addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runTask();
      e.stopPropagation();
    });

    document.querySelectorAll('.nav-item').forEach((b) => {
      b.onclick = () => setView(b.dataset.view);
    });

    $('flow').addEventListener('click', (e) => {
      const head = e.target.closest('.card-head');
      if (!head) return;
      const card = head.parentElement;
      const id = card.dataset.card;
      card.classList.toggle('open');
      if (card.classList.contains('open')) S.openCards.add(id);
      else S.openCards.delete(id);
    });

    $('fileTree').addEventListener('click', async (e) => {
      const dir = e.target.closest('[data-dir]');
      if (dir) {
        const box = document.querySelector(`[data-children="${CSS.escape(dir.dataset.dir)}"]`);
        if (box) box.classList.toggle('hidden');
        return;
      }
      const file = e.target.closest('[data-file]');
      if (!file) return;
      try {
        const d = await api('/api/file?path=' + encodeURIComponent(file.dataset.file));
        openModal(`
          <div class="modal-head"><span class="t mono">${esc(d.path)}</span>
            <button class="x" data-close>×</button></div>
          <div class="modal-body"><pre class="code">${esc(d.content)}</pre></div>`,
          { kind: 'file', closable: true });
        const m = document.querySelector('.modal');
        m.querySelector('[data-close]').onclick = () => closeModal();
      } catch (_) {}
    });

    $('snapList').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-rollback]');
      if (!btn) return;
      const name = btn.dataset.rollback;
      openModal(`
        <div class="modal-head"><span class="t">回滚到 ${esc(name)}？</span></div>
        <div class="modal-body">会用快照覆盖同名文件（不会删除快照之后新建的文件）。这是一次写操作，建议先再打一个快照。</div>
        <div class="modal-foot"><span class="note"></span>
          <button class="btn ghost" data-cancel>取消</button>
          <button class="btn danger" data-ok>确认回滚</button></div>`, { kind: 'rollback' });
      document.querySelector('.modal [data-cancel]').onclick = () => closeModal();
      document.querySelector('.modal [data-ok]').onclick = async () => {
        try {
          await api('/api/rollback', { method: 'POST', body: { snapshot: name } });
          toast('已回滚', 'ok');
          loadTree('.');
        } catch (_) {}
        closeModal();
      };
    });

    $('traceBody').addEventListener('click', (e) => {
      const tr = e.target.closest('[data-trace]');
      if (tr) openTrace(tr.dataset.trace);
    });

    $('skillGrid').addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-del-skill]');
      if (!btn) return;
      if (!window.confirm(`删除技能 skill_${btn.dataset.delSkill}？`)) return;
      await api('/api/skills?name=' + encodeURIComponent(btn.dataset.delSkill), { method: 'DELETE' });
      loadSkills();
    });

    $('replayPlay').onclick = toggleReplay;
    $('replayExit').onclick = exitReplay;
    $('replaySpeed').onclick = () => {
      if (!S.replay) return;
      S.replay.speed = S.replay.speed === 2 ? 1 : 2;
      $('replaySpeed').textContent = S.replay.speed + '×';
    };

    $('themeBtn').onclick = () => {
      const cur = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = cur;
      localStorage.setItem('fa.theme', cur);
    };

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modalCloser) {
        const overlay = document.querySelector('.overlay');
        if (overlay && overlay.dataset.kind === 'confirm') {
          const first = Object.values(S.pending)[0];
          if (first) {
            api('/api/confirm', {
              method: 'POST',
              body: { task_id: S.taskId, cid: first.cid, decision: 'deny', reason: '用户按 Esc 拒绝' },
            }).then(() => { delete S.pending[first.cid]; closeModal(); updateBadges(); });
          }
          return;
        }
        if (overlay && overlay.dataset.kind === 'action') return;
        closeModal();
      }
    });
  }

  // ------------------------------------------------------------------
  // 启动
  // ------------------------------------------------------------------
  async function boot() {
    const theme = localStorage.getItem('fa.theme');
    if (theme) document.documentElement.dataset.theme = theme;
    bind();
    try {
      S.config = await api('/api/config', { quiet: true });
      S.maxSteps = (S.config.limits && S.config.limits.max_steps) || 15;
      $('optMaxSteps').value = S.maxSteps;
      $('goal').placeholder = '告诉 Agent 要做什么，例如：统计工作区每个 .py 文件的行数并写入 stats.md（Ctrl+Enter 运行）';
      renderPanel();
      if (S.config.features && S.config.features.demo) {
        $('goal').value = '读取 config.yaml，把关键配置写成 notes.md，再用 python -V 确认环境';
        toast('演示模式：已填入示例目标，点「运行」即可看到完整交互', 'info', 6000);
      }
    } catch (e) {
      toast('读取配置失败，服务可能没启动', 'err');
    }
    loadTree('.');
    loadSnapshots();
    loadTrust();
    setStatus('idle');
  }

  boot();
})();
