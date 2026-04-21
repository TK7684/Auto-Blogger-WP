// PM2 config — one long-lived scheduler daemon replacing the old hot-loop.
//
// To deploy:
//   pm2 delete auto-blogger || true
//   pm2 start ecosystem.config.js
//   pm2 save
//
// The daemon runs src/scheduler.py, which has its own cron-style rules.
// Do NOT add --cron-restart here; the scheduler stays alive on purpose.
module.exports = {
  apps: [{
    name: "auto-blogger",
    cwd: "/home/tk578/Auto-Blogger-WP",
    script: "/home/tk578/Auto-Blogger-WP/.venv/bin/python3",
    args: "-m src.scheduler",
    interpreter: "none",
    autorestart: true,
    max_restarts: 10,
    min_uptime: "30s",
    restart_delay: 5000,
    max_memory_restart: "500M",
    env: {
      PYTHONUNBUFFERED: "1",
    },
    error_file: "/home/tk578/.pm2/logs/auto-blogger-error.log",
    out_file:   "/home/tk578/.pm2/logs/auto-blogger-out.log",
    merge_logs: true,
  }],
};
