"""Rewrite the ask-form to a modern all-in-one input box."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

html_path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

OLD_FORM = '''        <!-- Ask form -->
        <form class="ask-form" id="askForm">
          <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg,image/webp,application/pdf" hidden>
          <textarea
            id="questionInput"
            name="question"
            rows="2"
            maxlength="500"
            placeholder="Ask about train 12727, cancellation rules, Vijayawada station\u2026"
            required
            aria-label="Your question"
          ></textarea>
          <!-- File preview strip (shown when a file is attached) -->
          <div class="file-preview" id="filePreview" style="display:none;">
            <div class="file-preview-info">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              <span class="file-name" id="fileName"></span>
              <span class="file-size" id="fileSize"></span>
            </div>
            <button type="button" class="file-remove" id="fileRemove" title="Remove file">\u2715</button>
          </div>
          <div class="form-footer">
            <span class="char-count" id="charCount">0 / 500</span>
            <div class="form-actions">
              <kbd class="shortcut-hint">Ctrl + Enter</kbd>
              <button id="attachBtn" type="button" class="attach-btn" title="Attach image or PDF" aria-label="Attach file">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                </svg>
              </button>
              <button id="voiceBtn" type="button" class="voice-btn" title="Voice input (click to speak)" aria-label="Voice input">
                <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                <span class="voice-label">Mic</span>
              </button>
              <button id="submitQuestion" type="submit" class="ask-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                Ask
              </button>
            </div>
          </div>
        </form>'''

NEW_FORM = '''        <!-- Ask form -->
        <form class="ask-form" id="askForm">
          <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg,image/webp,application/pdf" hidden>
          <div class="input-box" id="inputBox">
            <!-- File preview (shown when file attached) -->
            <div class="file-preview" id="filePreview" style="display:none;">
              <div class="file-preview-info">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span class="file-name" id="fileName"></span>
                <span class="file-size" id="fileSize"></span>
              </div>
              <button type="button" class="file-remove" id="fileRemove" title="Remove file">\u2715</button>
            </div>
            <!-- Textarea -->
            <textarea
              id="questionInput"
              name="question"
              rows="2"
              maxlength="500"
              placeholder="Ask about train 12727, cancellation rules, Vijayawada station\u2026"
              required
              aria-label="Your question"
            ></textarea>
            <!-- Bottom toolbar inside the box -->
            <div class="input-toolbar">
              <div class="input-toolbar-left">
                <button id="attachBtn" type="button" class="tool-btn" title="Attach image or PDF" aria-label="Attach file">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="17" height="17">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                  </svg>
                </button>
                <button id="voiceBtn" type="button" class="tool-btn voice-btn" title="Voice input" aria-label="Voice input">
                  <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="17" height="17">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                  </svg>
                  <span class="voice-label">Mic</span>
                </button>
              </div>
              <div class="input-toolbar-right">
                <span class="char-count" id="charCount">0 / 500</span>
                <kbd class="shortcut-hint">Ctrl\u21b5</kbd>
                <button id="submitQuestion" type="submit" class="ask-btn">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                  Ask
                </button>
              </div>
            </div>
          </div>
        </form>'''

if OLD_FORM in html:
    html = html.replace(OLD_FORM, NEW_FORM)
    print('OK: form rewritten')
else:
    print('FAIL: form block not matched — check whitespace/encoding')
    idx = html.find('<!-- Ask form -->')
    print(repr(html[idx:idx+300]))

# Bump version
for v in ['v8','v7','v6','v5','v4','v3']:
    html = html.replace(f'styles.css?{v}', 'styles.css?v=9')
    html = html.replace(f'app.js?{v}', 'app.js?v=9')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('index.html written, v9')
