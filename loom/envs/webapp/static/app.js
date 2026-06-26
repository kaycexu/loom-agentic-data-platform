// 前端交互：所有动作走真实 /api/* HTTP 端点，成功后用 /state 重渲染。
// BrowserEnv 通过真实 DOM 操作（fill input 触发 change、click 按钮）驱动这些动作。

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function colLetter(cell) { return cell[0]; }
function rowNum(cell) { return parseInt(cell.slice(1), 10); }

// 根据 /state 的 cells 重建表格主体，使新行的 input 出现。
function renderSheet(state) {
  const cells = state.sheet.cells || {};
  let maxRow = 1;
  for (const k of Object.keys(cells)) {
    const n = rowNum(k);
    if (n > maxRow) maxRow = n;
  }
  const body = document.getElementById("sheet-body");
  let html = "";
  for (let n = 1; n <= maxRow + 1; n++) {
    const a = cells["A" + n] != null ? cells["A" + n] : "";
    const b = cells["B" + n] != null ? cells["B" + n] : "";
    if (n === 1) {
      html += `<tr data-row="1"><td class="rownum">1</td>` +
        `<td><span id="cell-A1">${a}</span></td>` +
        `<td><span id="cell-B1">${b}</span></td><td></td></tr>`;
    } else {
      html += `<tr data-row="${n}"><td class="rownum">${n}</td>` +
        `<td><input id="cell-A${n}" data-cell="A${n}" class="cell" value="${a}" /></td>` +
        `<td><input id="cell-B${n}" data-cell="B${n}" class="cell" value="${b}" /></td>` +
        `<td><button class="delete-row" id="delete-row-${n}" data-row="${n}" type="button">删除</button></td></tr>`;
    }
  }
  body.innerHTML = html;
  bindCellHandlers();
}

function renderEmail(state) {
  const el = document.getElementById("email-status");
  el.textContent = state.email.status;
  el.setAttribute("data-status", state.email.status);
}

function refresh(state) {
  renderSheet(state);
  renderEmail(state);
}

async function writeCell(cell, value) {
  const state = await post("/api/write_cell", { cell, value });
  refresh(state);
}

function bindCellHandlers() {
  document.querySelectorAll("input.cell").forEach((inp) => {
    inp.addEventListener("change", async () => {
      await writeCell(inp.dataset.cell, inp.value);
    });
  });
  document.querySelectorAll("button.delete-row").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const state = await post("/api/delete_row", { row: parseInt(btn.dataset.row, 10) });
      refresh(state);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("read-email").addEventListener("click", async () => {
    refresh(await post("/api/read_email", {}));
  });
  document.getElementById("mark-done").addEventListener("click", async () => {
    refresh(await post("/api/mark_email_done", {}));
  });
  bindCellHandlers();
});
