export default function LoadingSpinner({ size = 'md' }) {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-7 w-7'
  };

  return (
    <span
      className={`${sizes[size]} inline-block animate-spin rounded-full border-2 border-ink-700 border-t-accent`}
      aria-label="Loading"
    />
  );
}
