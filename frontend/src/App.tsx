import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import type { ActiveTab } from './components/Sidebar';
import { ImageUploadView } from './components/ImageUploadView';
import { PixelInspectorView } from './components/PixelInspectorView';
import { GrayscaleView } from './components/GrayscaleView';
import { StatisticsView } from './components/StatisticsView';
import { MatrixRankView } from './components/MatrixRankView';
import { EigenAnalysisView } from './components/EigenAnalysisView';
import { SVDView } from './components/SVDView';
import { ReconstructionView } from './components/ReconstructionView';
import { TransformationsView } from './components/TransformationsView';
import { PerturbationView } from './components/PerturbationView';
import type { ImageMetadata } from './types';
import { loadSampleImage, checkBackendHealth } from './services/api';

export function App() {
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [activeTab, setActiveTab] = useState<ActiveTab>('upload');
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // Sync dark mode class on <html>
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Check backend health and load initial sample image
  useEffect(() => {
    let isMounted = true;

    const pollHealth = async () => {
      const health = await checkBackendHealth();
      if (isMounted) {
        setBackendOnline(health.online);
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 10000);

    handleLoadSample();

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleLoadSample = async () => {
    setLoadingSample(true);
    try {
      const data = await loadSampleImage();
      setMetadata(data);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
      console.warn('Initial sample load deferred (backend server starting up or unreachable)');
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      <Header
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        metadata={metadata}
        onLoadSample={handleLoadSample}
        loadingSample={loadingSample}
        backendOnline={backendOnline}
      />

      <div className="flex-1 flex max-w-7xl w-full mx-auto">
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          hasImage={metadata !== null}
        />

        <main className="flex-1 p-6 md:p-8 overflow-y-auto">
          {activeTab === 'upload' && (
            <ImageUploadView
              metadata={metadata}
              onImageLoaded={(meta) => {
                setMetadata(meta);
                setBackendOnline(true);
                setActiveTab('rgb');
              }}
              onNavigateToRGB={() => setActiveTab('rgb')}
            />
          )}

          {metadata && activeTab === 'rgb' && (
            <PixelInspectorView metadata={metadata} />
          )}

          {metadata && activeTab === 'grayscale' && (
            <GrayscaleView metadata={metadata} />
          )}

          {metadata && activeTab === 'statistics' && (
            <StatisticsView metadata={metadata} darkMode={darkMode} />
          )}

          {metadata && activeTab === 'rank' && (
            <MatrixRankView metadata={metadata} darkMode={darkMode} />
          )}

          {metadata && activeTab === 'eigen' && (
            <EigenAnalysisView metadata={metadata} darkMode={darkMode} />
          )}

          {metadata && activeTab === 'svd' && (
            <SVDView
              metadata={metadata}
              darkMode={darkMode}
              onNavigateToReconstruction={() => setActiveTab('reconstruction')}
            />
          )}

          {metadata && activeTab === 'reconstruction' && (
            <ReconstructionView metadata={metadata} />
          )}

          {metadata && activeTab === 'transformations' && (
            <TransformationsView metadata={metadata} darkMode={darkMode} />
          )}

          {metadata && activeTab === 'perturbation' && (
            <PerturbationView metadata={metadata} darkMode={darkMode} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
