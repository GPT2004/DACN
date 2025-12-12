import React from 'react';

const Skeleton = ({
  width,
  height,
  circle = false,
  count = 1,
  className = '',
}) => {
  const skeletons = Array(count).fill(0);

  return (
    <>
      {skeletons.map((_, index) => (
        <div
          key={index}
          className={`
            bg-gray-200 animate-pulse
            ${circle ? 'rounded-full' : 'rounded'}
            ${className}
          `}
          style={{
            width: width || '100%',
            height: height || (circle ? width : '1rem'),
          }}
        />
      ))}
    </>
  );
};

// Preset Skeleton components
Skeleton.Text = ({ lines = 3, className = '' }) => (
  <div className={`space-y-2 ${className}`}>
    {Array(lines).fill(0).map((_, index) => (
      <Skeleton
        key={index}
        height="1rem"
        width={index === lines - 1 ? '80%' : '100%'}
      />
    ))}
  </div>
);

Skeleton.Card = ({ className = '' }) => (
  <div className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}>
    <Skeleton width="60%" height="1.5rem" className="mb-4" />
    <Skeleton.Text lines={3} />
  </div>
);

Skeleton.Avatar = ({ size = 'md', className = '' }) => {
  const sizes = {
    sm: 'w-8 h-8',
    md: 'w-10 h-10',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  };
  return <Skeleton circle width={sizes[size]} className={className} />;
};

export default Skeleton;
