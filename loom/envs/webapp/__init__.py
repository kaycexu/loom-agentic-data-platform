"""极简 Flask 应用 —— BrowserEnv 的真实「邮件+表格」后端。

单实例单状态（进程内存）。由 BrowserEnv 在子进程里以
`python -m loom.envs.webapp.app --port <port>` 启动，Playwright 通过真实 DOM
操作触发 /api/* 端点改变状态；状态可由 GET /state 读回（形状与 SheetEmailEnv
的 get_state() 完全一致）。
"""
