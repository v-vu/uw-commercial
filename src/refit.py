# --- Code Block ---
# Save the trained pipeline to a file, production model
with open('lasso_model_refit_07282025.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

