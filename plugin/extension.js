const vscode = require('vscode');

const API_HOST = 'http://0.0.0.0:2525';

const storage = {
    originalContent: '',
    refactoredContent: '',
    originalDocumentUriString: null,
    selectionToReplace: null,
};

class RefactorContentProvider {
    provideTextDocumentContent(uri) {
        if (uri.path === 'original') return storage.originalContent;
        if (uri.path === 'refactored') return storage.refactoredContent;
        return `Error: Could not find content for ${uri.path}`;
    }
}

function setRefactoringContext(visible) {
    vscode.commands.executeCommand('setContext', 'refactorPlugin.isRefactoring', visible);
}

function activate(context) {
    console.log('Congratulations, your "refactor-plugin-demo" is now active!');

    setRefactoringContext(false);

    const refactorProvider = new RefactorContentProvider();
    context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider('refactor', refactorProvider));

    let refactorCommand = vscode.commands.registerCommand('refactor-plugin.refactorCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selectedText = editor.document.getText(editor.selection);
        if (selectedText.length === 0) {
            vscode.window.showWarningMessage('Please select the code to refactor.');
            return;
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Calling Refactoring API...",
            cancellable: true
        }, async (progress, token) => {
            token.onCancellationRequested(() => {
                console.log("User canceled the refactoring operation.");
            });

            try {
                const { default: fetch } = await import('node-fetch');

                const response = await fetch(`${API_HOST}/refactor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: selectedText })
                });

                if (token.isCancellationRequested) return;

                if (!response.ok) {
                    const errorBody = await response.text();
                    throw new Error(`API request failed with status ${response.status}: ${errorBody}`);
                }

                const data = await response.json();

                storage.originalDocumentUriString = editor.document.uri.toString();
                storage.selectionToReplace = editor.selection;
                storage.originalContent = selectedText;
                storage.refactoredContent = data.refactored_code;

                const originalUri = vscode.Uri.parse('refactor:original');
                const refactoredUri = vscode.Uri.parse('refactor:refactored');
                const diffTitle = `Refactor Suggestion for ${editor.document.fileName.split('/').pop()}`;

                await vscode.commands.executeCommand('vscode.diff', originalUri, refactoredUri, diffTitle);

                setRefactoringContext(true);

            } catch (error) {
                console.error("Refactoring failed:", error);
                const errorMessage = error.message.includes('ECONNREFUSED')
                    ? "Could not connect to the API. Is the local service running?"
                    : `Refactoring failed: ${error.message}`;
                vscode.window.showErrorMessage(errorMessage);
            }
        });
    });

    let applyCommand = vscode.commands.registerCommand('refactor-plugin.applyChanges', async () => {
        const { originalDocumentUriString, selectionToReplace, refactoredContent } = storage;
        if (!originalDocumentUriString || !selectionToReplace) {
            vscode.window.showErrorMessage('Could not apply changes. Context was lost.');
            return setRefactoringContext(false);
        }

        const workspaceEdit = new vscode.WorkspaceEdit();
        workspaceEdit.replace(vscode.Uri.parse(originalDocumentUriString), selectionToReplace, refactoredContent);

        const success = await vscode.workspace.applyEdit(workspaceEdit);
        if (success) {
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
            vscode.window.showInformationMessage('Changes applied successfully!');
        } else {
            vscode.window.showErrorMessage('Failed to apply changes.');
        }
        setRefactoringContext(false);
    });

    let rejectCommand = vscode.commands.registerCommand('refactor-plugin.rejectChanges', async () => {
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
        setRefactoringContext(false);
        vscode.window.showInformationMessage('Refactoring rejected.');
    });

    context.subscriptions.push(refactorCommand, applyCommand, rejectCommand);
}

function deactivate() {
    setRefactoringContext(false);
}

module.exports = {
    activate,
    deactivate
};
