# GitHub Authentication Setup

## Quick Fix: Use Personal Access Token

### 1. Create Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "CFR Implementation"
4. Check: ✅ `repo` (full control)
5. Click "Generate token"
6. **COPY THE TOKEN** (save it somewhere safe!)

### 2. Update Git Remote

```bash
cd "/Users/zhangwei/Desktop/CS590_Reinforcement Learning/Final Project/Poker_ReinforcementLearning"

# Replace YOUR_TOKEN with your actual token
git remote set-url origin https://YOUR_TOKEN@github.com/davidwang0116/Poker_ReinforcementLearning.git

# Verify
git remote -v

# Now push
git push -u origin feature/cfr-implementation
```

### Example:
If your token is `ghp_abc123xyz`, run:
```bash
git remote set-url origin https://ghp_abc123xyz@github.com/davidwang0116/Poker_ReinforcementLearning.git
```

## Alternative: Use GitHub CLI (Easiest!)

If you have GitHub CLI installed:

```bash
# Login
gh auth login

# Follow prompts:
# - GitHub.com
# - HTTPS
# - Authenticate with browser

# Then push
git push -u origin feature/cfr-implementation
```

## Troubleshooting

### "Invalid username or token"
- Make sure you copied the entire token
- Token should start with `ghp_`
- Make sure you selected `repo` scope when creating

### "Permission denied"
- Check if you have write access to the repo
- Make sure you're using the correct GitHub username

### "Repository not found"
- Verify the repo URL: https://github.com/davidwang0116/Poker_ReinforcementLearning
- Make sure you have access to this repository

## Security Note

**Never commit tokens to git!** The token is only used in the remote URL, which is stored locally in `.git/config` (not tracked by git).

## After Successful Push

Once you've pushed successfully, you can:

1. Go to: https://github.com/davidwang0116/Poker_ReinforcementLearning
2. You should see your branch: `feature/cfr-implementation`
3. Create a Pull Request if needed
4. Share with your partners!
