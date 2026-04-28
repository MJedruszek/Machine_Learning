import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to store charts
os.makedirs('charts', exist_ok=True)

# Read the CSV file
df = pd.read_csv('wyniki_modeli.csv')  # Replace with your actual file path

# Convert Param_Value to numeric where possible
df['Param_Value_numeric'] = pd.to_numeric(df['Param_Value'], errors='coerce')

# Filter for tree-based models with max_depth
tree_models = ['CustomTree', 'SklearnTree', 'RandomForrest']
df_trees = df[df['Model'].isin(tree_models) & (df['Param_Name'] == 'max_depth')].copy()
df_trees['max_depth'] = df_trees['Param_Value_numeric']

# ============================================
# CHART 1: Line Chart - Accuracy vs max_depth (separate lines for each model)
# ============================================
plt.figure(figsize=(12, 8))
for test_size in sorted(df_trees['Test_Size'].unique()):
    subset = df_trees[df_trees['Test_Size'] == test_size]
    for model in tree_models:
        model_data = subset[subset['Model'] == model]
        if not model_data.empty:
            plt.plot(model_data['max_depth'], model_data['Accuracy'], 
                    marker='o', linewidth=2, markersize=8, label=f"{model} (test={test_size})")
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs max_depth - Trees & RandomForest', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([5, 10, 20])
plt.tight_layout()
plt.savefig('charts/chart1_accuracy_vs_maxdepth.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart1_accuracy_vs_maxdepth.png")

# ============================================
# CHART 2: Bar Chart - All models compared (Test Size = 0.2, max_depth=5)
# ============================================
plt.figure(figsize=(12, 8))
depth5_data = df[df['Test_Size'] == 0.2].copy()
depth5_trees = df_trees[(df_trees['Test_Size'] == 0.2) & (df_trees['max_depth'] == 5)]
other_models = depth5_data[~depth5_data['Model'].isin(tree_models)]

bar_data = pd.concat([
    depth5_trees[['Model', 'Accuracy']],
    other_models[['Model', 'Accuracy']]
])
bar_data = bar_data.drop_duplicates(subset=['Model'])

bars = plt.bar(range(len(bar_data)), bar_data['Accuracy'].values, color='skyblue', edgecolor='black')
plt.xticks(range(len(bar_data)), bar_data['Model'].values, rotation=45, ha='right')
plt.ylabel('Accuracy', fontsize=12)
plt.title('All Models Comparison (Test Size=0.2, max_depth=5 for trees)', fontsize=14)
plt.ylim(0.85, 0.95)
for bar, val in zip(bars, bar_data['Accuracy'].values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('charts/chart2_all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart2_all_models_comparison.png")

# ============================================
# CHART 3: Grouped Bar Chart - Accuracy by Test Size (max_depth=5)
# ============================================
plt.figure(figsize=(12, 8))
depth5_all = df_trees[df_trees['max_depth'] == 5]
pivot_data = depth5_all.pivot_table(index='Model', columns='Test_Size', values='Accuracy')
pivot_data.plot(kind='bar', edgecolor='black')
plt.title('Trees & Forest at max_depth=5 by Test Size', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.legend(title='Test Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0.85, 0.95)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/chart3_grouped_bar_by_testsize.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart3_grouped_bar_by_testsize.png")

# ============================================
# CHART 4: Heatmap - Accuracy across Test Size and max_depth
# ============================================
plt.figure(figsize=(10, 8))
heatmap_data = df_trees.groupby(['Test_Size', 'max_depth'])['Accuracy'].mean().unstack()
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
            cbar_kws={'label': 'Accuracy'}, linewidths=2, linecolor='white')
plt.title('Average Accuracy Heatmap (All Trees + RandomForest)', fontsize=14)
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('Test Size', fontsize=12)
plt.tight_layout()
plt.savefig('charts/chart4_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart4_heatmap.png")

# ============================================
# CHART 5: Line Chart - Overfitting trend (all models at Test Size=0.2)
# ============================================
plt.figure(figsize=(12, 8))
test02_data = df_trees[df_trees['Test_Size'] == 0.2]
for model in tree_models:
    model_data = test02_data[test02_data['Model'] == model]
    if not model_data.empty:
        plt.plot(model_data['max_depth'], model_data['Accuracy'], 
                marker='s', linewidth=2, markersize=10, label=model)
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Overfitting Trend (Test Size = 0.2)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([5, 10, 20])
constant_acc = df[df['Test_Size'] == 0.2]['Accuracy'].iloc[0]
plt.axhline(y=constant_acc, color='red', linestyle='--', linewidth=2, 
           label=f'Logistic Regression/MLP ({constant_acc:.4f})')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('charts/chart5_overfitting_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart5_overfitting_trend.png")

# ============================================
# CHART 6: Bar Chart - Performance drop from depth=5 to depth=20
# ============================================
plt.figure(figsize=(12, 8))
depth5_data = df_trees[df_trees['max_depth'] == 5].set_index(['Model', 'Test_Size'])['Accuracy']
depth20_data = df_trees[df_trees['max_depth'] == 20].set_index(['Model', 'Test_Size'])['Accuracy']
performance_drop = depth5_data - depth20_data
performance_drop = performance_drop.reset_index()
performance_drop.columns = ['Model', 'Test_Size', 'Drop']
pivot_drop = performance_drop.pivot(index='Model', columns='Test_Size', values='Drop')
pivot_drop.plot(kind='bar', edgecolor='black', color=['lightcoral', 'salmon', 'lightgreen'])
plt.title('Performance Drop (depth=5 to depth=20)', fontsize=14)
plt.ylabel('Accuracy Drop (lower is better)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.legend(title='Test Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/chart6_performance_drop.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart6_performance_drop.png")

# ============================================
# CHART 7: Additional - Accuracy comparison across all test sizes (line plot)
# ============================================
plt.figure(figsize=(12, 8))
for model in tree_models:
    model_data = df_trees[df_trees['Model'] == model]
    for test_size in sorted(df_trees['Test_Size'].unique()):
        subset = model_data[model_data['Test_Size'] == test_size]
        if not subset.empty:
            plt.plot(subset['max_depth'], subset['Accuracy'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f"{model} (test={test_size})", 
                    linestyle='-', alpha=0.7)
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison Across Test Sizes', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks([5, 10, 20])
plt.tight_layout()
plt.savefig('charts/chart7_model_comparison_across_testsizes.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart7_model_comparison_across_testsizes.png")

# ============================================
# CHART 8: Box plot - Distribution of accuracy by model
# ============================================
plt.figure(figsize=(12, 8))
df_all_models = pd.concat([
    df_trees[['Model', 'Accuracy']],
    df[df['Model'].isin(['LogisticRegression', 'MLP'])][['Model', 'Accuracy']]
])
sns.boxplot(data=df_all_models, x='Model', y='Accuracy', palette='Set2')
plt.title('Accuracy Distribution by Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('charts/chart8_accuracy_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: chart8_accuracy_distribution.png")