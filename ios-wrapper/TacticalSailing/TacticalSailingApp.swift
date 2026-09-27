import SwiftUI

/// Entry point for the thin iOS shell around the hosted Streamlit app.
@main
struct TacticalSailingApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .ignoresSafeArea(.container, edges: .bottom)
        }
    }
}

struct ContentView: View {
    /// Replace with your deployed Streamlit URL (must be https).
    /// NOTE: streamlit.app subdomains can resolve to *someone else's* app if
    /// you forget this step -- don't ship without setting your real URL.
    private let appURL = URL(string: "https://replace-me.invalid/?embed=true")!

    var body: some View {
        WebView(url: appURL)
    }
}
