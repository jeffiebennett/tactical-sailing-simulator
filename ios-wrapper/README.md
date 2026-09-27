# iOS wrapper

A thin SwiftUI app that loads the hosted Streamlit game in a `WKWebView`. The
Python game logic is unchanged; this is only a native shell around it.

Already built and verified in the iOS 27 Simulator on this machine:
- `TacticalSailing/TacticalSailingApp.swift` and `WebView.swift` — the app.
- `project.yml` — the [xcodegen](https://github.com/yonaskolb/XcodeGen) spec.
- `TacticalSailing.xcodeproj` — generated from `project.yml` (git-ignored;
  regenerate any time with `xcodegen generate`).

Confirmed working: builds clean, loads a real page over HTTPS correctly, and
shows a friendly fallback ("Can't reach the race server") when the URL can't
be resolved.

## 1. Host the Streamlit app
1. Push this repo to GitHub (`.venv` and the generated `.xcodeproj` are
   git-ignored).
2. On https://share.streamlit.io, create an app from that repo with main
   file `main.py`.
3. Copy the resulting `https://<name>.streamlit.app` URL.

## 2. Point the app at it
Edit `TacticalSailing/TacticalSailingApp.swift` and set `appURL` to your real
URL, keeping `?embed=true`. **Do this before running** — the placeholder
`replace-me.invalid` is deliberately non-resolving; an earlier placeholder
(`YOUR-APP-NAME.streamlit.app`) turned out to resolve to a stranger's
unrelated live Streamlit app, so don't reuse patterns like that.

If you change `project.yml` (bundle ID, deployment target, etc.), regenerate
with `xcodegen generate` before building.

## 3. Run
```
open TacticalSailing.xcodeproj
```
- **Simulator:** pick any iPhone simulator and press Run — no signing needed.
- **Your own iPhone:** plug it in, select it as the destination, and set your
  Apple ID under Signing & Capabilities > Team (free accounts work; the
  install expires after 7 days and needs re-running from Xcode to renew).
- **App Store distribution:** needs a paid Apple Developer account ($99/yr).

Or from the command line:
```
xcodebuild -project TacticalSailing.xcodeproj -scheme TacticalSailing \
  -destination 'platform=iOS Simulator,name=iPhone 17' build
```

## Notes
- The app needs a network connection; the game logic runs on the Streamlit
  server, not on the phone.
- Streamlit Community Cloud apps sleep when idle, so the first launch may
  take ~30 seconds.
