document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  if (!form) return;
  form.addEventListener("submit", (event) => {
    const fields = form.querySelectorAll('input[type="number"]');
    for (const field of fields) {
      const value = Number(field.value);
      if (field.value.trim() === "" || !Number.isFinite(value) || value < 0 || value > 10) {
        event.preventDefault();
        field.focus();
        field.setCustomValidity("Enter a number between 0 and 10.");
        field.reportValidity();
        return;
      }
      field.setCustomValidity("");
    }
  });
  form.querySelectorAll('input[type="number"]').forEach((field) => {
    field.addEventListener("input", () => field.setCustomValidity(""));
  });
});